from datetime import datetime
from model.heuristic import AdvancedCyclePredictor
from core.parse_data import DataProcessor
import json
def display_menu():
    print("\n" + "=" * 40)
    print("HEALTH TRACKER - MENSTRUAL CYCLE PREDICTION")
    print("=" * 40)
    print("1. Record your last period")
    print("2. Predict your next period")
    print("3. View your cycle history")
    print("4. Clear all records")
    print("5. Exit")
    print("=" * 40)

def record_period(data_processor):
    print("\n[ RECORD YOUR LAST PERIOD ]")
    start_date = input("Start date (DD-MM-YYYY): ")
    end_date = input("End date (DD-MM-YYYY): ")
    symptoms = input("Symptoms (comma separated): ")
    # Treat 'none' (case-insensitive) as no symptoms
    if symptoms.strip().lower() == 'none':
        symptoms = ''
    try:
        cycle_length = int(input("Average cycle length (days): "))
        entry = data_processor.parse_data(start_date, end_date, symptoms, cycle_length)
        data_processor.save_data(entry)
        print("\n✅ Record saved successfully!")
    except ValueError:
        print("⚠️ Invalid input! Please enter a valid number for cycle length.")
    except Exception as e:
        print(f"⚠️ Error: {str(e)}")

def predict_next_period():
    print("\n[ PREDICT YOUR NEXT PERIOD ]")
    processor = DataProcessor()
    predictor = AdvancedCyclePredictor(processor.DATA_FILE, processor.SYMPTOMS_FILE)
    
    if not predictor.data:
        print("⚠️ No data available. Please record your period first.")
        return
    
    prediction = predictor.predict_next_cycle()
    
    print("\n📅 PREDICTION RESULTS:")
    print(f"Start date: {prediction['predicted_start']}")
    print(f"End date: {prediction['predicted_end']}")
    print(f"Duration: {prediction['predicted_period_length']} days")
    print(f"Confidence: {prediction['confidence'].capitalize()}")
    
    if prediction['predicted_symptoms']:
        print("\n⚠️ Possible symptoms:")
        for symptom in prediction['predicted_symptoms']:
            print(f" - {symptom}")

def view_history():
    print("\n[ YOUR CYCLE HISTORY ]")
    processor = DataProcessor()
    predictor = AdvancedCyclePredictor(processor.DATA_FILE, processor.SYMPTOMS_FILE)
    
    if not predictor.data:
        print("No records found.")
        return
    
    history = predictor.get_cycle_history()
    
    for i, record in enumerate(history, 1):
        print(f"\nCYCLE #{i}")
        print(f"Start: {record['start_date']}")
        print(f"End: {record['end_date']}")
        print(f"Duration: {record['length']} days")
        print(f"Cycle length: {record['cycle_length']} days")
        
        if record['symptoms']:
            print("Symptoms: " + ", ".join(record['symptoms']))

def clear_records():
    confirm = input("⚠️ Are you sure you want to delete ALL records? (y/n): ")
    if confirm.lower() == 'y':
        processor = DataProcessor()
        with open(processor.DATA_FILE, "w") as f:
            json.dump([], f)
        print("✅ All records have been deleted.")

def main():
    processor = DataProcessor()
    
    while True:
        display_menu()
        choice = input("Select an option (1-5): ")
        
        if choice == '1':
            record_period(processor)
        elif choice == '2':
            predict_next_period()
        elif choice == '3':
            view_history()
        elif choice == '4':
            clear_records()
        elif choice == '5':
            print("Thank you for using Health Tracker!")
            break
        else:
            print("⚠️ Invalid choice. Please select 1-5.")

if __name__ == "__main__":
    main()