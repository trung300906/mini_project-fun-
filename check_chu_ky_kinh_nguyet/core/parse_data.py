import datetime as dt
import json
import os

class DataProcessor:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    DATA_FILE = os.path.join(DATA_DIR, 'user_data.json')
    SYMPTOMS_FILE = os.path.join(DATA_DIR, 'symptoms_dict.json')
    
    @staticmethod
    def calculate_length(start_date: str, end_date: str) -> int:
        start = dt.datetime.strptime(start_date, "%d-%m-%Y")
        end = dt.datetime.strptime(end_date, "%d-%m-%Y")
        return (end - start).days
    
    @staticmethod
    def parse_symptoms(symptoms_str: str) -> list:
        return [symptom.strip() for symptom in symptoms_str.split(",") if symptom.strip()]
    
    def symptoms_to_numeric_vector(self, symptom_list: list) -> list:
        with open(self.SYMPTOMS_FILE, "r") as f:
            symptom_dict = json.load(f)
        vector = [0] * len(symptom_dict)
        for symptom in symptom_list:
            if symptom in symptom_dict:
                vector[symptom_dict[symptom]] = 1
        return vector
    
    def parse_data(self, start_date: str, end_date: str, symptoms_str: str, cycle_length: int) -> dict:
        symptoms_parse = self.parse_symptoms(symptoms_str)
        print(f"Parsed symptoms: {symptoms_parse}")
        symptoms_vector = self.symptoms_to_numeric_vector(symptoms_parse)
        length = self.calculate_length(start_date, end_date)
        return {
            "start_date": start_date,
            "end_date": end_date,
            "length": length,
            "symptoms": symptoms_vector,
            "cycle_length": cycle_length,
            "created_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def save_data(self, entry: dict):
        if not os.path.exists(self.DATA_DIR):
            os.makedirs(self.DATA_DIR)
        
        if not os.path.exists(self.DATA_FILE):
            with open(self.DATA_FILE, "w") as f:
                json.dump([], f, indent=2)
        
        with open(self.DATA_FILE, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
        
        data.append(entry)
        
        with open(self.DATA_FILE, "w") as f:
            json.dump(data, f, indent=2)