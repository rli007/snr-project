import json
import os
from typing import List, Dict
import re
import time

class RAGSystem:
    def __init__(self, base_dir: str = "/Users/RyanWorks/desktop/ap-data-by-period"):
        self.base_dir = base_dir
        self.chunks = []
        self.period_chunks = {}  # store chunks by period
        self.exam_info_chunks = []  # store exam info chunks

        # load all chunks
        self.load_data()
    
    def preprocess_text(self, text: str) -> List[str]:
        """convert text to lowercase and split into words"""
        words = re.findall(r'\b\w+\b', text.lower())
        return words
    
    def load_data(self):
        """load all chunks from period and exam info dirs"""
        # load period data
        for period_num in range(1, 10):
            period_dir = os.path.join(self.base_dir, f"period{period_num}_data")
            period_file = os.path.join(period_dir, f"period_{period_num}_chunks.json")
            
            if os.path.exists(period_file):
                try:
                    with open(period_file, 'r', encoding='utf-8') as f:
                        period_chunks = json.load(f)
                        self.period_chunks[str(period_num)] = period_chunks
                        self.chunks.extend(period_chunks)
                except Exception as e:
                    print(f"Error loading period {period_num}: {str(e)}")
        
        # load exam info
        exam_info_dir = os.path.join(self.base_dir, "exam_info_data")
        exam_info_file = os.path.join(exam_info_dir, "exam_info_chunks.json")
        
        if os.path.exists(exam_info_file):
            try:
                with open(exam_info_file, 'r', encoding='utf-8') as f:
                    self.exam_info_chunks = json.load(f)
                    self.chunks.extend(self.exam_info_chunks)
            except Exception as e:
                print(f"Error loading exam info: {str(e)}")
    
    def get_relevant_chunks(self, query: str, top_k: int = 3) -> List[Dict]:
        """get most relevant chunks for a query using keyword matching"""
        if not self.chunks:
            return []
        
        # process query
        query_words = set(self.preprocess_text(query))
        
        # score each chunk based on keyword matches
        chunk_scores = []
        for chunk in self.chunks:
            chunk_text = chunk["text"].lower()
            chunk_words = set(self.preprocess_text(chunk_text))
            
            # calculate relevance score
            # 1. number of matching keywords
            matching_keywords = query_words.intersection(chunk_words)
            score = len(matching_keywords)
            
            # 2. boost score for exact phrase matches
            for word in query_words:
                if word in chunk_text:
                    score += 1
            
            # 3. boost score for chunks from same period if query mentions a period
            period_mentions = [f"period {i}" for i in range(1, 10) if f"period {i}" in query.lower()]
            if period_mentions and any(period in chunk["metadata"].get("period_title", "").lower() for period in period_mentions):
                score *= 1.5
            
            # 4. boost score for exam info chunks if query is about exam format/scoring
            exam_keywords = {"exam", "test", "score", "grading", "rubric", "format", "multiple choice", "dbq", "saq", "leq"}
            if any(keyword in query.lower() for keyword in exam_keywords) and chunk["metadata"].get("section") == "Exam Information":
                score *= 1.5
            
            chunk_scores.append((chunk, score))
        
        # sort chunks by score and get top k
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, score in chunk_scores[:top_k] if score > 0]
    
    def format_context(self, chunks: List[Dict]) -> str:
        """format retrieved chunks into a context string"""
        context = "Relevant information from AP US History CED:\n\n"
        
        for chunk in chunks:
            if chunk["metadata"].get("section") == "Exam Information":
                context += "From Exam Information:\n"
            else:
                period = chunk["metadata"].get("period", "Unknown")
                period_title = chunk["metadata"].get("period_title", f"Period {period}")
                context += f"From Period {period} ({period_title}):\n"
            
            context += chunk["text"] + "\n\n"
        
        return context 