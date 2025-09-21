import os
import io
import re
import pickle
import numpy as np
import streamlit as st
from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pdfplumber
from collections import Counter, defaultdict
import nltk
from textstat import flesch_reading_ease, flesch_kincaid_grade
import docx
import pandas as pd
from datetime import datetime
import json
import hashlib
import torch
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

# Hugging Face imports
try:
    from transformers import (
        pipeline, 
        AutoTokenizer, 
        AutoModel,
        AutoModelForSequenceClassification,
        AutoModelForQuestionAnswering,
        T5ForConditionalGeneration,
        T5Tokenizer,
        BartForConditionalGeneration,
        BartTokenizer
    )
    HF_AVAILABLE = True
except ImportError:
    st.error("⚠️ Hugging Face Transformers not installed. Please install: pip install transformers sentence-transformers torch")
    HF_AVAILABLE = False

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag

# Enhanced Configuration
CHUNK_SIZE_CHARS = 512  # Optimized for transformer models
CHUNK_OVERLAP_CHARS = 50
TOP_K = 5
MIN_SIMILARITY_THRESHOLD = 0.3  # Higher threshold for better results
MAX_LENGTH = 512  # For transformer models

class HuggingFaceDocumentChatbot:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.corpus = []
        self.embeddings = None
        self.is_trained = False
        self.document_summary = ""
        self.key_topics = []
        self.document_type = ""
        self.document_metadata = {}
        self.conversation_context = []
        self.response_history = []
        self.stop_words = set(stopwords.words('english'))
        
        # Initialize models
        self.models = self._initialize_models()
    
    @st.cache_resource
    def _initialize_models(_self):
        """Initialize and cache Hugging Face models"""
        if not HF_AVAILABLE:
            return None
            
        try:
            models = {}
            
            with st.spinner("🤖 Loading AI models... This may take a few minutes on first run."):
                # Sentence transformer for semantic embeddings
                st.info("Loading semantic understanding model...")
                models['embedder'] = SentenceTransformer('all-MiniLM-L6-v2')
                
                # Question answering model
                st.info("Loading question-answering model...")
                models['qa_pipeline'] = pipeline(
                    "question-answering",
                    model="distilbert-base-cased-distilled-squad",
                    device=0 if torch.cuda.is_available() else -1
                )
                
                # Text summarization model
                st.info("Loading text summarization model...")
                models['summarizer'] = pipeline(
                    "summarization",
                    model="facebook/bart-large-cnn",
                    device=0 if torch.cuda.is_available() else -1,
                    max_length=150,
                    min_length=50
                )
                
                # Text classification for document type
                st.info("Loading document classification model...")
                models['classifier'] = pipeline(
                    "zero-shot-classification",
                    model="facebook/bart-large-mnli",
                    device=0 if torch.cuda.is_available() else -1
                )
                
                # Text generation for responses
                st.info("Loading text generation model...")
                models['generator'] = pipeline(
                    "text2text-generation",
                    model="google/flan-t5-base",
                    device=0 if torch.cuda.is_available() else -1,
                    max_length=200
                )
                
                # Sentiment analysis
                st.info("Loading sentiment analysis model...")
                models['sentiment'] = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    device=0 if torch.cuda.is_available() else -1
                )
                
                # Named Entity Recognition
                st.info("Loading entity recognition model...")
                models['ner'] = pipeline(
                    "ner",
                    model="dbmdz/bert-large-cased-finetuned-conll03-english",
                    aggregation_strategy="simple",
                    device=0 if torch.cuda.is_available() else -1
                )
                
            st.success("✅ All AI models loaded successfully!")
            return models
            
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            return None
    
    def classify_document_type(self, text: str, filename: str) -> str:
        """Use AI to classify document type"""
        try:
            if self.models and 'classifier' in self.models:
                candidate_labels = [
                    "academic research paper",
                    "business document", 
                    "technical documentation",
                    "legal document",
                    "financial report",
                    "medical document",
                    "educational material",
                    "news article",
                    "manual or guide",
                    "general document"
                ]
                
                # Use first 1000 chars for classification
                text_sample = text[:1000]
                result = self.models['classifier'](text_sample, candidate_labels)
                
                # Get the top classification
                top_label = result['labels'][0]
                confidence = result['scores'][0]
                
                return f"{top_label.title()} (Confidence: {confidence:.2f})"
            else:
                return self._fallback_document_type(filename, text)
        except Exception as e:
            st.warning(f"AI classification failed: {e}")
            return self._fallback_document_type(filename, text)
    
    def _fallback_document_type(self, filename: str, text: str) -> str:
        """Fallback document classification"""
        filename_lower = filename.lower()
        content_sample = text[:2000].lower()
        
        if filename_lower.endswith('.pdf'):
            doc_type = "PDF Document"
        elif filename_lower.endswith('.docx'):
            doc_type = "Word Document"
        elif filename_lower.endswith('.txt'):
            doc_type = "Text Document"
        elif filename_lower.endswith(('.csv', '.xlsx')):
            doc_type = "Data Document"
        else:
            doc_type = "Document"
        
        # Simple keyword-based classification
        if any(word in content_sample for word in ['abstract', 'methodology', 'conclusion']):
            return f"Academic {doc_type}"
        elif any(word in content_sample for word in ['revenue', 'profit', 'business']):
            return f"Business {doc_type}"
        else:
            return doc_type
    
    def extract_entities_with_ai(self, text: str) -> Dict[str, List[str]]:
        """Extract entities using AI NER model"""
        try:
            if self.models and 'ner' in self.models:
                # Use NER model
                entities = self.models['ner'](text[:2000])  # Limit for performance
                
                organized_entities = {
                    'people': [],
                    'organizations': [],
                    'locations': [],
                    'miscellaneous': []
                }
                
                for entity in entities:
                    label = entity['entity_group'].upper()
                    word = entity['word']
                    
                    if label in ['PER', 'PERSON']:
                        organized_entities['people'].append(word)
                    elif label in ['ORG', 'ORGANIZATION']:
                        organized_entities['organizations'].append(word)
                    elif label in ['LOC', 'LOCATION']:
                        organized_entities['locations'].append(word)
                    else:
                        organized_entities['miscellaneous'].append(word)
                
                # Clean and deduplicate
                for key in organized_entities:
                    organized_entities[key] = list(set([
                        item.strip() for item in organized_entities[key] 
                        if len(item.strip()) > 1
                    ]))[:10]
                
                return organized_entities
            else:
                return self._fallback_entity_extraction(text)
        except Exception as e:
            st.warning(f"AI entity extraction failed: {e}")
            return self._fallback_entity_extraction(text)
    
    def _fallback_entity_extraction(self, text: str) -> Dict[str, List[str]]:
        """Fallback entity extraction using simple NLP"""
        entities = {
            'people': [],
            'organizations': [],
            'locations': [],
            'dates': [],
            'numbers': []
        }
        
        # Extract dates using regex
        date_pattern = r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b\s*\d{1,2},?\s*\d{2,4})\b'
        entities['dates'] = list(set(re.findall(date_pattern, text, re.IGNORECASE)))[:10]
        
        # Extract numbers
        number_pattern = r'\b\d+(?:\.\d+)?(?:%|\s*percent|million|billion|thousand)?\b'
        entities['numbers'] = list(set(re.findall(number_pattern, text)))[:10]
        
        return entities
    
    def generate_ai_summary(self, text: str) -> str:
        """Generate summary using AI summarization model"""
        try:
            if self.models and 'summarizer' in self.models and len(text) > 100:
                # Split text into chunks if too long
                max_chunk_length = 1000
                if len(text) > max_chunk_length:
                    chunks = [text[i:i+max_chunk_length] for i in range(0, len(text), max_chunk_length)]
                    summaries = []
                    
                    for chunk in chunks[:3]:  # Limit to 3 chunks for performance
                        if len(chunk.strip()) > 50:
                            result = self.models['summarizer'](chunk, max_length=50, min_length=20)
                            summaries.append(result[0]['summary_text'])
                    
                    return " ".join(summaries)
                else:
                    result = self.models['summarizer'](text, max_length=100, min_length=30)
                    return result[0]['summary_text']
            else:
                return self._fallback_summary(text)
        except Exception as e:
            st.warning(f"AI summarization failed: {e}")
            return self._fallback_summary(text)
    
    def _fallback_summary(self, text: str) -> str:
        """Fallback summary generation"""
        sentences = sent_tokenize(text)
        if len(sentences) <= 3:
            return text
        
        # Simple extractive summary - take first and last sentences plus one from middle
        summary_sentences = [sentences[0]]
        if len(sentences) > 2:
            summary_sentences.append(sentences[len(sentences)//2])
        summary_sentences.append(sentences[-1])
        
        return " ".join(summary_sentences)
    
    def analyze_text_complexity(self, text: str) -> Dict:
        """Enhanced text complexity analysis with AI sentiment"""
        try:
            reading_ease = flesch_reading_ease(text)
            grade_level = flesch_kincaid_grade(text)
            
            # Determine difficulty level
            if reading_ease >= 90:
                difficulty = "Very Easy"
                audience = "Elementary School"
            elif reading_ease >= 80:
                difficulty = "Easy"
                audience = "Middle School"
            elif reading_ease >= 70:
                difficulty = "Fairly Easy"
                audience = "High School"
            elif reading_ease >= 60:
                difficulty = "Standard"
                audience = "College Level"
            elif reading_ease >= 50:
                difficulty = "Fairly Difficult"
                audience = "Graduate Level"
            elif reading_ease >= 30:
                difficulty = "Difficult"
                audience = "Academic/Professional"
            else:
                difficulty = "Very Difficult"
                audience = "Expert Level"
            
            # Get sentiment analysis
            sentiment_result = {"sentiment": "neutral", "confidence": 0.5}
            try:
                if self.models and 'sentiment' in self.models:
                    sample_text = text[:500]  # Use sample for sentiment
                    sentiment_analysis = self.models['sentiment'](sample_text)
                    sentiment_result = {
                        "sentiment": sentiment_analysis[0]['label'].lower(),
                        "confidence": sentiment_analysis[0]['score']
                    }
            except:
                pass
            
            # Additional metrics
            sentences = sent_tokenize(text)
            words = word_tokenize(text)
            
            avg_sentence_length = len(words) / len(sentences) if sentences else 0
            
            return {
                "reading_ease": round(reading_ease, 1),
                "grade_level": round(grade_level, 1),
                "difficulty": difficulty,
                "audience": audience,
                "avg_sentence_length": round(avg_sentence_length, 1),
                "total_words": len(words),
                "total_sentences": len(sentences),
                "sentiment": sentiment_result["sentiment"],
                "sentiment_confidence": round(sentiment_result["confidence"], 2)
            }
        except:
            return {
                "reading_ease": 0, 
                "grade_level": 0, 
                "difficulty": "Unknown",
                "audience": "Unknown",
                "avg_sentence_length": 0,
                "total_words": 0,
                "total_sentences": 0,
                "sentiment": "neutral",
                "sentiment_confidence": 0.5
            }
    
    def extract_text_from_file(self, uploaded_file) -> Tuple[str, Dict]:
        """Extract text from various file formats"""
        file_extension = uploaded_file.name.lower().split('.')[-1]
        metadata = {
            'filename': uploaded_file.name,
            'file_size': uploaded_file.size,
            'file_type': file_extension
        }
        
        try:
            if file_extension == 'pdf':
                return self.extract_from_pdf(uploaded_file.getvalue()), metadata
            elif file_extension == 'docx':
                return self.extract_from_docx(uploaded_file), metadata
            elif file_extension == 'txt':
                return str(uploaded_file.read(), 'utf-8'), metadata
            elif file_extension == 'csv':
                return self.extract_from_csv(uploaded_file), metadata
            elif file_extension == 'xlsx':
                return self.extract_from_excel(uploaded_file), metadata
            else:
                content = str(uploaded_file.read(), 'utf-8', errors='ignore')
                return content, metadata
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            return "", metadata
    
    def extract_from_pdf(self, pdf_bytes: bytes) -> str:
        """Extract text from PDF"""
        full_text = ""
        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        full_text += text + "\n"
        except Exception as e:
            st.error(f"Error extracting PDF text: {str(e)}")
        return full_text
    
    def extract_from_docx(self, docx_file) -> str:
        """Extract text from Word document"""
        try:
            doc = docx.Document(docx_file)
            full_text = ""
            for paragraph in doc.paragraphs:
                full_text += paragraph.text + "\n"
            return full_text
        except Exception as e:
            st.error(f"Error extracting DOCX text: {str(e)}")
            return ""
    
    def extract_from_csv(self, csv_file) -> str:
        """Extract and summarize CSV content"""
        try:
            df = pd.read_csv(csv_file)
            summary = f"CSV Data Analysis:\n"
            summary += f"Dataset contains {len(df)} rows and {len(df.columns)} columns.\n"
            summary += f"Column names: {', '.join(df.columns.tolist())}\n\n"
            
            # Add sample data
            summary += "Sample data preview:\n"
            summary += df.head().to_string() + "\n\n"
            
            # Add statistics for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                summary += "Statistical summary of numeric columns:\n"
                summary += df[numeric_cols].describe().to_string()
            
            return summary
        except Exception as e:
            st.error(f"Error processing CSV: {str(e)}")
            return ""
    
    def extract_from_excel(self, excel_file) -> str:
        """Extract and summarize Excel content"""
        try:
            df = pd.read_excel(excel_file)
            summary = f"Excel Data Analysis:\n"
            summary += f"Spreadsheet contains {len(df)} rows and {len(df.columns)} columns.\n"
            summary += f"Column headers: {', '.join(df.columns.tolist())}\n\n"
            summary += "Data preview:\n"
            summary += df.head().to_string()
            return summary
        except Exception as e:
            st.error(f"Error processing Excel file: {str(e)}")
            return ""
    
    def create_semantic_chunks(self, text: str) -> List[str]:
        """Create semantic chunks optimized for transformers"""
        if not text or len(text.strip()) == 0:
            return []
        
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > CHUNK_SIZE_CHARS and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk = (current_chunk + " " + sentence).strip()
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Filter out very short chunks
        return [chunk for chunk in chunks if len(chunk.strip()) > 50]
    
    def build_corpus(self, uploaded_file) -> bool:
        """Build AI-powered corpus from uploaded file"""
        try:
            full_text, metadata = self.extract_text_from_file(uploaded_file)
            
            if not full_text or len(full_text.strip()) < 100:
                st.error("The uploaded file doesn't contain sufficient text content for AI analysis")
                return False
            
            # Store metadata
            self.document_metadata = metadata
            
            # AI-powered document analysis
            with st.spinner("🤖 AI is analyzing document type and content..."):
                self.document_type = self.classify_document_type(full_text, uploaded_file.name)
                complexity = self.analyze_text_complexity(full_text)
                entities = self.extract_entities_with_ai(full_text)
                ai_summary = self.generate_ai_summary(full_text)
            
            # Create semantic chunks
            chunks = self.create_semantic_chunks(full_text)
            
            if not chunks:
                st.error("Could not create meaningful text chunks for AI processing")
                return False
            
            # Create embeddings using sentence transformer
            if self.models and 'embedder' in self.models:
                with st.spinner("🧠 Creating AI embeddings for semantic understanding..."):
                    self.embeddings = self.models['embedder'].encode(chunks)
                    st.success(f"✅ Created semantic embeddings for {len(chunks)} text segments")
            
            # Build corpus with AI enhancements
            self.corpus = []
            for i, chunk in enumerate(chunks):
                self.corpus.append({
                    'id': i,
                    'text': chunk,
                    'length': len(chunk),
                    'sentences': len(sent_tokenize(chunk))
                })
            
            # Generate document summary
            self.document_summary = f"""
🤖 **AI-Powered Document Analysis**
**Document Type**: {self.document_type}
**AI Summary**: {ai_summary}
**Complexity**: {complexity['difficulty']} ({complexity['audience']})
**Sentiment**: {complexity['sentiment'].title()} (Confidence: {complexity['sentiment_confidence']})
**Statistics**: {complexity['total_words']} words, {complexity['total_sentences']} sentences
**Reading Level**: Grade {complexity['grade_level']} (Flesch Score: {complexity['reading_ease']})
"""
            
            if any(entities.values()):
                entity_summary = []
                for entity_type, entity_list in entities.items():
                    if entity_list:
                        entity_summary.append(f"{entity_type.title()}: {len(entity_list)}")
                if entity_summary:
                    self.document_summary += f"\n**AI-Detected Entities**: {', '.join(entity_summary)}"
            
            self.is_trained = True
            
            st.success(f"🎉 AI analysis complete! Processed {len(self.corpus)} semantic chunks with {len(self.embeddings) if self.embeddings is not None else 'basic'} embeddings")
            return True
            
        except Exception as e:
            st.error(f"Error during AI analysis: {str(e)}")
            return False
    
    def semantic_search(self, query: str, top_k: int = TOP_K) -> List[Dict]:
        """Perform semantic search using AI embeddings"""
        if not self.is_trained or not query.strip():
            return []
        
        try:
            if self.models and 'embedder' in self.models and self.embeddings is not None:
                # Use semantic search with embeddings
                query_embedding = self.models['embedder'].encode([query])
                
                # Calculate cosine similarity
                similarities = cosine_similarity(query_embedding, self.embeddings).flatten()
                
                # Get top k results
                top_indices = similarities.argsort()[-top_k:][::-1]
                
                results = []
                for idx in top_indices:
                    score = similarities[idx]
                    if score >= MIN_SIMILARITY_THRESHOLD:
                        result = self.corpus[idx].copy()
                        result['score'] = float(score)
                        result['similarity_type'] = 'semantic'
                        results.append(result)
                
                return results
            else:
                # Fallback to traditional TF-IDF search
                return self.fallback_search(query, top_k)
                
        except Exception as e:
            st.error(f"Error during semantic search: {str(e)}")
            return self.fallback_search(query, top_k)
    
    def fallback_search(self, query: str, top_k: int) -> List[Dict]:
        """Fallback search using TF-IDF"""
        try:
            vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
            corpus_texts = [item['text'] for item in self.corpus]
            tfidf_matrix = vectorizer.fit_transform(corpus_texts)
            query_vector = vectorizer.transform([query])
            
            similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
            top_indices = similarities.argsort()[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                score = similarities[idx]
                if score >= 0.1:  # Lower threshold for fallback
                    result = self.corpus[idx].copy()
                    result['score'] = float(score)
                    result['similarity_type'] = 'keyword'
                    results.append(result)
            
            return results
        except:
            return []
    
    def generate_ai_response(self, query: str, context: str) -> str:
        """Generate response using AI models"""
        try:
            if self.models:
                # Try question-answering first
                if 'qa_pipeline' in self.models and len(context) > 50:
                    try:
                        qa_result = self.models['qa_pipeline'](
                            question=query,
                            context=context[:2000]  # Limit context length
                        )
                        
                        if qa_result['score'] > 0.1:  # Confidence threshold
                            return f"🤖 **AI Answer**: {qa_result['answer']}\n\n**Confidence**: {qa_result['score']:.2f}"
                    except:
                        pass
                
                # Try text generation as alternative
                if 'generator' in self.models:
                    try:
                        prompt = f"Based on this context: {context[:800]}\n\nQuestion: {query}\n\nAnswer:"
                        result = self.models['generator'](prompt, max_length=150, do_sample=True)
                        
                        generated_text = result[0]['generated_text']
                        # Clean up the response
                        if "Answer:" in generated_text:
                            answer = generated_text.split("Answer:")[-1].strip()
                            return f"🤖 **AI Response**: {answer}"
                    except:
                        pass
            
            # Fallback to rule-based response
            return self.generate_fallback_response(query, context)
            
        except Exception as e:
            st.warning(f"AI response generation failed: {e}")
            return self.generate_fallback_response(query, context)
    
    def generate_fallback_response(self, query: str, context: str) -> str:
        """Generate fallback response when AI models fail"""
        sentences = sent_tokenize(context)
        
        # Find sentences most relevant to the query
        query_words = set(query.lower().split())
        relevant_sentences = []
        
        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            overlap = len(query_words.intersection(sentence_words))
            if overlap > 0:
                relevant_sentences.append((sentence, overlap))
        
        # Sort by relevance and take top 3
        relevant_sentences.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [sent[0] for sent in relevant_sentences[:3]]
        
        if top_sentences:
            response = "📋 **Based on the document content:**\n\n"
            for i, sentence in enumerate(top_sentences, 1):
                response += f"{i}. {sentence}\n\n"
            return response
        else:
            return "❌ I couldn't find relevant information in the document to answer your question."
    
    def generate_answer(self, query: str, top_k: int = TOP_K) -> str:
        """Main function to generate AI-powered responses"""
        if not query.strip():
            return "Please ask me a question about the document."
        
        # Perform semantic search
        results = self.semantic_search(query, top_k)
        
        if not results:
            return f"🤔 I couldn't find relevant information about '{query}' in the document.\n\nTry asking about: {', '.join(self.get_suggested_topics()[:5])}"
        
        # Combine context from search results
        context = " ".join([result['text'] for result in results])
        
        # Generate AI response
        ai_response = self.generate_ai_response(query, context)
        
        # Add metadata
        search_type = results[0].get('similarity_type', 'unknown')
        avg_score = np.mean([result['score'] for result in results])
        
        metadata = f"\n\n---\n"
        metadata += f"**Search Method**: {search_type.title()}\n"
        metadata += f"**Confidence**: {avg_score:.2f}\n"
        metadata += f"**Sources**: {len(results)} relevant sections"
        
        # Store in conversation history
        self.conversation_context.append({
            'query': query,
            'results_count': len(results),
            'confidence': avg_score,
            'timestamp': datetime.now()
        })
        
        return ai_response + metadata
    
    def get_suggested_topics(self) -> List[str]:
        """Get suggested topics from document"""
        if not self.corpus:
            return []
        
        # Simple keyword extraction from first few chunks
        all_text = " ".join([chunk['text'] for chunk in self.corpus[:3]])
        words = word_tokenize(all_text.lower())
        
        # Filter meaningful words
        meaningful_words = [
            word for word in words 
            if word.isalpha() and word not in self.stop_words and len(word) > 4
        ]
        
        # Get most common words
        word_freq = Counter(meaningful_words)
        return [word for word, _ in word_freq.most_common(10)]
    
    def get_document_insights(self) -> str:
        """Get AI-powered document insights"""
        if not self.is_trained:
            return "No document has been uploaded and analyzed yet."
        
        insights = [self.document_summary]
        
        # Add AI-powered suggestions based on document type
        if "academic" in self.document_type.lower():
            insights.append("\n🎓 **AI Suggestions for Academic Document:**")
            insights.extend([
                "• What is the main research question?",
                "• What methodology was used?",
                "• What are the key findings?",
                "• What conclusions were drawn?"
            ])
        elif "business" in self.document_type.lower():
            insights.append("\n💼 **AI Suggestions for Business Document:**")
            insights.extend([
                "• What are the main objectives?",
                "• What strategies are proposed?",
                "• What are the financial implications?",
                "• What recommendations are made?"
            ])
        elif "technical" in self.document_type.lower():
            insights.append("\n🔧 **AI Suggestions for Technical Document:**")
            insights.extend([
                "• What are the system requirements?",
                "• How does the implementation work?",
                "• What are the technical specifications?",
                "• Are there any troubleshooting steps?",
                "• What are the configuration options?"
            ])
        elif "legal" in self.document_type.lower():
            insights.append("\n⚖️ **AI Suggestions for Legal Document:**")
            insights.extend([
                "• What are the key terms and conditions?",
                "• What are the rights and obligations?",
                "• Are there any important dates or deadlines?",
                "• What are the penalties or consequences?"
            ])
        elif "financial" in self.document_type.lower():
            insights.append("\n💰 **AI Suggestions for Financial Document:**")
            insights.extend([
                "• What are the financial performance metrics?",
                "• What are the revenue and expense trends?",
                "• Are there any risk factors mentioned?",
                "• What are the future projections?"
            ])
        else:
            insights.append("\n💡 **AI-Generated Question Suggestions:**")
            suggested_topics = self.get_suggested_topics()
            if suggested_topics:
                for topic in suggested_topics[:5]:
                    insights.append(f"• What does the document say about {topic}?")
        
        # Add conversation statistics if available
        if self.conversation_context:
            insights.append(f"\n📊 **Conversation Statistics:**")
            insights.append(f"• Total questions asked: {len(self.conversation_context)}")
            avg_confidence = np.mean([ctx['confidence'] for ctx in self.conversation_context])
            insights.append(f"• Average answer confidence: {avg_confidence:.2f}")
            
        return "\n".join(insights)
    
    def get_conversation_history(self) -> List[Dict]:
        """Get conversation history with enhanced metadata"""
        return self.conversation_context
    
    def clear_conversation_history(self):
        """Clear conversation history"""
        self.conversation_context.clear()
        self.response_history.clear()
    
    def export_conversation(self) -> Dict:
        """Export conversation and document analysis for download"""
        export_data = {
            'document_metadata': self.document_metadata,
            'document_type': self.document_type,
            'document_summary': self.document_summary,
            'conversation_history': [
                {
                    'query': ctx['query'],
                    'results_count': ctx['results_count'],
                    'confidence': ctx['confidence'],
                    'timestamp': ctx['timestamp'].isoformat()
                }
                for ctx in self.conversation_context
            ],
            'export_timestamp': datetime.now().isoformat(),
            'total_chunks': len(self.corpus),
            'ai_models_used': list(self.models.keys()) if self.models else []
        }
        return export_data


def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="🤖 AI-Powered Document Chatbot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better UI
    st.markdown("""
    <style>
    .main {
        padding-top: 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #4CAF50;
        background-color: #f0f2f6;
    }
    .user-message {
        border-left-color: #2196F3;
        background-color: #e3f2fd;
    }
    .ai-response {
        border-left-color: #FF9800;
        background-color: #fff3e0;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = HuggingFaceDocumentChatbot()
        st.session_state.chat_history = []
    
    # Header
    st.title("🤖 AI-Powered Document Chatbot")
    st.markdown("*Advanced document analysis and question-answering using Hugging Face Transformers*")
    
    # Sidebar
    with st.sidebar:
        st.header("📄 Document Upload")
        
        uploaded_file = st.file_uploader(
            "Choose a document file",
            type=['pdf', 'docx', 'txt', 'csv', 'xlsx'],
            help="Supported formats: PDF, Word, Text, CSV, Excel"
        )
        
        if uploaded_file is not None:
            st.success(f"File uploaded: {uploaded_file.name}")
            
            if st.button("🚀 Analyze Document with AI", type="primary"):
                with st.spinner("🤖 AI is processing your document..."):
                    success = st.session_state.chatbot.build_corpus(uploaded_file)
                    if success:
                        st.success("✅ Document analysis complete!")
                        st.rerun()
        
        # Document insights
        if st.session_state.chatbot.is_trained:
            st.header("📊 Document Insights")
            insights = st.session_state.chatbot.get_document_insights()
            st.markdown(insights)
            
            # Export functionality
            st.header("💾 Export")
            if st.button("📥 Export Analysis"):
                export_data = st.session_state.chatbot.export_conversation()
                st.download_button(
                    label="Download Analysis JSON",
                    data=json.dumps(export_data, indent=2),
                    file_name=f"document_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            
            if st.button("🗑️ Clear Conversation"):
                st.session_state.chatbot.clear_conversation_history()
                st.session_state.chat_history.clear()
                st.success("Conversation cleared!")
                st.rerun()
    
    # Main chat interface
    if st.session_state.chatbot.is_trained:
        st.header("💬 Ask Questions About Your Document")
        
        # Display conversation history
        if st.session_state.chat_history:
            st.subheader("📜 Conversation History")
            for i, (user_msg, ai_response) in enumerate(st.session_state.chat_history):
                # User message
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>🙋 You:</strong> {user_msg}
                </div>
                """, unsafe_allow_html=True)
                
                # AI response
                st.markdown(f"""
                <div class="chat-message ai-response">
                    <strong>🤖 AI Assistant:</strong><br>
                    {ai_response.replace('\n', '<br>')}
                </div>
                """, unsafe_allow_html=True)
        
        # Query input
        st.markdown("---")
        query = st.text_input(
            "Ask a question about the document:",
            placeholder="e.g., What is the main topic? What are the key findings?",
            key="query_input"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            ask_button = st.button("🚀 Ask AI", type="primary")
        with col2:
            if st.button("💡 Get AI Suggestions"):
                suggestions = st.session_state.chatbot.get_suggested_topics()[:5]
                if suggestions:
                    st.info(f"**Try asking about:** {', '.join(suggestions)}")
        
        if ask_button and query:
            with st.spinner("🤖 AI is thinking..."):
                response = st.session_state.chatbot.generate_answer(query)
                st.session_state.chat_history.append((query, response))
                st.rerun()
        
        # Quick actions
        st.markdown("---")
        st.subheader("⚡ Quick Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📋 Summarize Document"):
                with st.spinner("Creating summary..."):
                    response = st.session_state.chatbot.generate_answer("Provide a comprehensive summary of this document")
                    st.session_state.chat_history.append(("Summarize Document", response))
                    st.rerun()
        
        with col2:
            if st.button("🔍 Key Topics"):
                with st.spinner("Extracting key topics..."):
                    response = st.session_state.chatbot.generate_answer("What are the main topics and themes in this document?")
                    st.session_state.chat_history.append(("Key Topics", response))
                    st.rerun()
        
        with col3:
            if st.button("📈 Important Facts"):
                with st.spinner("Finding important facts..."):
                    response = st.session_state.chatbot.generate_answer("What are the most important facts, figures, or statistics mentioned?")
                    st.session_state.chat_history.append(("Important Facts", response))
                    st.rerun()
        
        # Performance metrics
        if st.session_state.chatbot.conversation_context:
            st.markdown("---")
            st.subheader("📊 Performance Metrics")
            
            metrics = st.session_state.chatbot.conversation_context
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_confidence = np.mean([ctx['confidence'] for ctx in metrics])
                st.metric("Average Confidence", f"{avg_confidence:.2f}")
            
            with col2:
                total_queries = len(metrics)
                st.metric("Total Queries", total_queries)
            
            with col3:
                semantic_queries = sum(1 for ctx in metrics if ctx.get('search_type') == 'semantic')
                st.metric("AI Semantic Searches", semantic_queries)
    
    else:
        # Welcome screen
        st.header("🎯 Welcome to AI-Powered Document Analysis!")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### 🚀 Advanced Features:
            
            **🤖 State-of-the-Art AI Models:**
            - **Semantic Understanding**: Advanced sentence transformers for meaning-based search
            - **Question Answering**: BERT-based models for precise answers
            - **Document Classification**: AI-powered document type detection
            - **Text Summarization**: BART models for intelligent summaries
            - **Named Entity Recognition**: Extract people, organizations, locations
            - **Sentiment Analysis**: Understand document tone and sentiment
            
            **📊 Intelligent Analysis:**
            - AI-powered document type classification
            - Automatic complexity and readability assessment
            - Smart entity extraction and organization
            - Semantic chunking for better understanding
            - Context-aware response generation
            
            **💬 Natural Conversations:**
            - Ask questions in natural language
            - Get AI-generated answers with confidence scores
            - Semantic search finds relevant content even with different words
            - Conversation history and analytics
            - Export your analysis and conversations
            
            **📁 Multiple File Formats:**
            - PDF documents
            - Word documents (.docx)
            - Text files (.txt)
            - CSV data files
            - Excel spreadsheets (.xlsx)
            """)
        
        with col2:
            st.markdown("""
            ### 🎓 How to Get Started:
            
            1. **📄 Upload** a document using the sidebar
            2. **🤖 Analyze** - Let AI process your document
            3. **💬 Ask** questions in natural language
            4. **🔍 Explore** AI-powered insights and suggestions
            5. **📥 Export** your analysis when done
            
            ### 🛠️ AI Models Used:
            - **all-MiniLM-L6-v2**: Semantic embeddings
            - **DistilBERT**: Question answering  
            - **BART-Large-CNN**: Text summarization
            - **BART-Large-MNLI**: Document classification
            - **FLAN-T5-Base**: Text generation
            - **RoBERTa**: Sentiment analysis
            - **BERT-Large**: Named entity recognition
            """)
        
        # Demo section
        st.markdown("---")
        st.header("🎬 Try It Out!")
        st.info("👆 Upload a document in the sidebar to start your AI-powered document analysis experience!")


if __name__ == "__main__":
    main()