import os
import json
from typing import List, Dict
import re
from datetime import datetime
import PyPDF2

class PeriodProcessor:
    def __init__(self, period_file: str, output_dir: str = "period_data"):
        self.period_file = period_file
        self.output_dir = output_dir
        self.chunks = []
        
        # make output dir if it doesnt exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def read_period(self) -> str:
        """read the period pdf file"""
        try:
            text = ""
            with open(self.period_file, 'rb') as file:
                # create pdf reader
                pdf_reader = PyPDF2.PdfReader(file)
                
                # get text from each page
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    # clean up weird chars and spaces
                    page_text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', page_text)
                    # add spaces between words that are stuck together
                    page_text = re.sub(r'([a-z])([A-Z])', r'\1 \2', page_text)
                    # add spaces after periods and commas
                    page_text = re.sub(r'([.,!?])([A-Za-z])', r'\1 \2', page_text)
                    # fix extra spaces
                    page_text = re.sub(r'\s+', ' ', page_text)
                    text += page_text + "\n"
            
            return text
                
        except Exception as e:
            print(f"Error reading period file: {str(e)}")
            raise
    
    def split_into_chunks(self, text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        """split text into overlapping chunks"""
        # split by subsections
        subsections = re.split(r'\n(?=[A-Z][A-Za-z\s]+:)', text)
        
        chunks = []
        
        for subsection in subsections:
            # clean the text
            subsection = subsection.strip()
            if not subsection:
                continue
            
            # if subsection is small enough, keep it as one chunk
            if len(subsection) <= chunk_size:
                chunks.append(subsection)
                continue
            
            # otherwise split into smaller chunks with overlap
            sentences = re.split(r'(?<=[.!?])\s+', subsection)
            current_chunk = []
            current_size = 0
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                # if adding this sentence would make chunk too big
                # save current chunk and start new one
                if current_size + len(sentence) > chunk_size and current_chunk:
                    chunks.append(' '.join(current_chunk))
                    # keep last few sentences for overlap
                    overlap_sentences = current_chunk[-2:]  # keep last 2 sentences
                    current_chunk = overlap_sentences
                    current_size = sum(len(s) + 1 for s in current_chunk)
                
                current_chunk.append(sentence)
                current_size += len(sentence) + 1  # +1 for space
            
            # add any leftover sentences as final chunk
            if current_chunk:
                chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def process_chunk(self, chunk: str, index: int) -> Dict:
        """process a single chunk into structured format"""
        # get period number from filename
        period_match = re.search(r'p(\d+)', self.period_file)
        period = period_match.group(1) if period_match else "Unknown"
        
        # try to get period title from chunk
        period_title_match = re.search(r'Period \d+:\s*([^\n]+)', chunk)
        period_title = period_title_match.group(1).strip() if period_title_match else f"Period {period}"
        
        # create metadata
        metadata = {
            "chunk_id": index,
            "period": period,
            "period_title": period_title,
            "timestamp": datetime.now().isoformat(),
            "source": f"AP US History Period {period}"
        }
        
        return {
            "text": chunk,
            "metadata": metadata
        }
    
    def save_chunks(self, chunks: List[Dict]):
        """save processed chunks to json files"""
        # get period number from filename
        period_match = re.search(r'p(\d+)', self.period_file)
        period = period_match.group(1) if period_match else "Unknown"
        
        # save chunks to a single file
        output_file = os.path.join(self.output_dir, f"period_{period}_chunks.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2)

    def process(self):
        """main processing function"""
        print(f"Reading period file: {self.period_file}")
        text = self.read_period()
        
        print("Splitting into chunks...")
        raw_chunks = self.split_into_chunks(text)
        
        print("Processing chunks...")
        processed_chunks = [self.process_chunk(chunk, i) for i, chunk in enumerate(raw_chunks)]
        
        print("Saving chunks...")
        self.save_chunks(processed_chunks)
        
        print(f"Processing complete! {len(processed_chunks)} chunks created.")
        print(f"Output saved to {self.output_dir}/")

class ExamInfoProcessor(PeriodProcessor):
    def process_chunk(self, chunk: str, index: int) -> Dict:
        """process a single chunk for exam info"""
        # create metadata
        metadata = {
            "chunk_id": index,
            "section": "Exam Information",
            "timestamp": datetime.now().isoformat(),
            "source": "AP US History Exam Information"
        }
        
        return {
            "text": chunk,
            "metadata": metadata
        }
    
    def save_chunks(self, chunks: List[Dict]):
        """save processed chunks to json files"""
        output_file = os.path.join(self.output_dir, "exam_info_chunks.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2)

def process_period(period_num: int):
    """process a single period pdf"""
    period_file = f"/Users/RyanWorks/downloads/p{period_num}.pdf"
    output_dir = f"/Users/RyanWorks/desktop/ap-data-by-period/period{period_num}_data"
    
    print(f"\nProcessing Period {period_num}...")
    processor = PeriodProcessor(period_file, output_dir)
    processor.process()

def process_exam_info():
    """process the exam info pdf"""
    exam_file = "/Users/RyanWorks/downloads/examinformation.pdf"
    output_dir = "/Users/RyanWorks/desktop/ap-data-by-period/exam_info_data"
    
    print("\nProcessing Exam Information...")
    processor = ExamInfoProcessor(exam_file, output_dir)
    processor.process()

def main():
    # process all periods 1-9
    for period_num in range(1, 10):
        try:
            process_period(period_num)
        except Exception as e:
            print(f"Error processing period {period_num}: {str(e)}")
            continue
    
    # process exam info
    try:
        process_exam_info()
    except Exception as e:
        print(f"Error processing exam information: {str(e)}")
    
    print("\nAll content processed!")
