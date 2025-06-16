import pandas as pd
import numpy as np
import torch
import json
import os
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import random

class MedicalTriageDataGenerator:
    """
    Synthetic medical triage data generator for testing and augmentation.
    """
    
    def __init__(self, seed=42):
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        
        # Medical knowledge base for realistic data generation
        self.symptoms_database = {
            'chest_pain': ['göğüs ağrısı', 'chest pain', 'cardiac pain'],
            'abdominal_pain': ['karın ağrısı', 'abdominal pain', 'stomach pain'],
            'headache': ['baş ağrısı', 'headache', 'migraine'],
            'fever': ['ateş', 'fever', 'high temperature'],
            'shortness_of_breath': ['nefes darlığı', 'dyspnea', 'breathing difficulty'],
            'trauma': ['travma', 'injury', 'accident'],
            'neurological': ['nörolojik', 'neurological', 'seizure']
        }
        
        self.comorbidities = [
            'diabetes', 'hypertension', 'heart disease', 'asthma', 'copd',
            'kidney disease', 'liver disease', 'cancer', 'stroke history'
        ]
        
        self.trauma_types = [
            'travma_kafa', 'travma_gogus', 'travma_batin', 'travma_kol',
            'travma_bacak', 'travma_el', 'travma_ayak', 'travma_boyun', 'travma_parmak'
        ]
    
    def generate_synthetic_dataset(self, num_samples=1000, save_path='triaj_data.csv'):
        """
        Generate a complete synthetic medical triage dataset.
        
        Args:
            num_samples (int): Number of synthetic patients to generate
            save_path (str): Path to save the generated dataset
        
        Returns:
            pd.DataFrame: Generated synthetic dataset
        """
        print(f"Generating {num_samples} synthetic medical triage records...")
        
        # Initialize data structure
        data = {
            'yaş': [],
            'cinsiyet': [],
            'sistolik kb': [],
            'diastolik kb': [],
            'solunum sayısı': [],
            'nabız': [],
            'ateş': [],
            'saturasyon': [],
            'ek hastalıklar': [],
            'semptomlar_non travma_genel 01': [],
            'semptomlar_non travma_genel 02': [],
            'göz': [],
            'göğüs ağrısı': [],
            'karın ağrısı': [],
            'travma_ayak': [],
            'travma_bacak': [],
            'travma_batin': [],
            'travma_boyun': [],
            'travma_el': [],
            'travma_gogus': [],
            'travma_goz': [],
            'travma_kafa': [],
            'travma_kol': [],
            'travma_parmak': [],
            'diğer travmalar': [],
            'dahiliye hastalıklar': [],
            'psikiyatri': [],
            'kardiyoloji': [],
            'göğüs hastalıkları': [],
            'nörolojik hastalıklar': [],
            'beyin cerrahi': [],
            'kalp damar cerrahisi': [],
            'kbb': [],
            'göz hastalıkları': [],
            'İntaniye': [],
            'Üroloji': [],
            'Çevresel ve toksikoloji acilleri': [],
            'kadın ve doğum hastalıkları': [],
            'genel cerrahi hastalıklar': [],
            'deri hastalıkları': [],
            'diğer diyagnoz_travma': [],
            'diğer diyagnoz': [],
            'triyaj alanı': [],
            'doğru triyaj': [],
            'created': []
        }
        
        for i in range(num_samples):
            # Generate patient demographics
            age = self._generate_age()
            gender = self._generate_gender()
            
            # Generate vital signs based on triage severity
            triage_level = self._determine_triage_level()
            vitals = self._generate_vitals(age, triage_level)
            
            # Generate symptoms and conditions
            symptoms = self._generate_symptoms(triage_level)
            comorbidities = self._generate_comorbidities(age)
            
            # Generate trauma if applicable
            trauma_data = self._generate_trauma(triage_level)
            
            # Generate medical specialties
            specialties = self._generate_specialties(symptoms, trauma_data)
            
            # Generate timestamp
            timestamp = self._generate_timestamp()
            
            # Populate data
            data['yaş'].append(age)
            data['cinsiyet'].append(gender)
            data['sistolik kb'].append(vitals['systolic'])
            data['diastolik kb'].append(vitals['diastolic'])
            data['solunum sayısı'].append(vitals['respiratory_rate'])
            data['nabız'].append(vitals['heart_rate'])
            data['ateş'].append(vitals['temperature'])
            data['saturasyon'].append(vitals['oxygen_saturation'])
            
            # Symptoms and conditions
            data['ek hastalıklar'].append(comorbidities)
            data['semptomlar_non travma_genel 01'].append(symptoms.get('general_01', ''))
            data['semptomlar_non travma_genel 02'].append(symptoms.get('general_02', ''))
            data['göz'].append(symptoms.get('eye', ''))
            data['göğüs ağrısı'].append(symptoms.get('chest_pain', ''))
            data['karın ağrısı'].append(symptoms.get('abdominal_pain', ''))
            
            # Trauma data - ensure all trauma types are populated
            for trauma_type in ['travma_ayak', 'travma_bacak', 'travma_batin', 'travma_boyun',
                              'travma_el', 'travma_gogus', 'travma_goz', 'travma_kafa',
                              'travma_kol', 'travma_parmak']:
                data[trauma_type].append(trauma_data.get(trauma_type, ''))
            
            data['diğer travmalar'].append(trauma_data.get('other_trauma', ''))
            
            # Medical specialties - ensure all specialties are populated
            for specialty in ['dahiliye hastalıklar', 'psikiyatri', 'kardiyoloji',
                            'göğüs hastalıkları', 'nörolojik hastalıklar', 'beyin cerrahi',
                            'kalp damar cerrahisi', 'kbb', 'göz hastalıkları', 'İntaniye',
                            'Üroloji', 'Çevresel ve toksikoloji acilleri',
                            'kadın ve doğum hastalıkları', 'genel cerrahi hastalıklar',
                            'deri hastalıkları']:
                data[specialty].append(specialties.get(specialty, ''))
            
            data['diğer diyagnoz_travma'].append(specialties.get('other_trauma_diagnosis', ''))
            data['diğer diyagnoz'].append(specialties.get('other_diagnosis', ''))
            
            # Triage levels
            triage_area = self._map_triage_to_area(triage_level)
            data['triyaj alanı'].append(triage_area)
            data['doğru triyaj'].append(triage_area)  # Assume initial triage is correct
            data['created'].append(timestamp)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Save to file
        df.to_csv(save_path, index=False)
        print(f"✅ Synthetic dataset saved to {save_path}")
        print(f"📊 Dataset statistics:")
        print(f"   - Total samples: {len(df)}")
        print(f"   - Age range: {df['yaş'].min()}-{df['yaş'].max()}")
        print(f"   - Gender distribution: {df['cinsiyet'].value_counts().to_dict()}")
        print(f"   - Triage distribution: {df['doğru triyaj'].value_counts().to_dict()}")
        
        return df
    
    def _generate_age(self):
        """Generate realistic age distribution."""
        # Weighted age distribution for emergency department
        age_ranges = [(0, 18, 0.15), (18, 35, 0.25), (35, 55, 0.30), (55, 75, 0.20), (75, 95, 0.10)]
        
        for min_age, max_age, weight in age_ranges:
            if np.random.random() < weight:
                return np.random.randint(min_age, max_age + 1)
        
        return np.random.randint(18, 65)  # Default adult range
    
    def _generate_gender(self):
        """Generate gender with realistic distribution."""
        return np.random.choice(['Erkek', 'Kadın'], p=[0.48, 0.52])
    
    def _determine_triage_level(self):
        """Determine triage level with realistic ED distribution."""
        # Realistic emergency department triage distribution
        return np.random.choice([0, 1, 2], p=[0.60, 0.30, 0.10])  # Green, Yellow, Red
    
    def _generate_vitals(self, age, triage_level):
        """Generate vital signs based on age and triage severity."""
        # Base vital signs
        if age < 18:  # Pediatric
            base_hr = 100
            base_rr = 20
            base_temp = 36.5
            base_sat = 98
            base_sys = 100
            base_dia = 60
        elif age > 65:  # Elderly
            base_hr = 75
            base_rr = 16
            base_temp = 36.2
            base_sat = 96
            base_sys = 140
            base_dia = 80
        else:  # Adult
            base_hr = 80
            base_rr = 16
            base_temp = 36.8
            base_sat = 98
            base_sys = 120
            base_dia = 70
        
        # Adjust based on triage level
        if triage_level == 2:  # Red (Critical)
            hr_mult = np.random.uniform(1.2, 1.8)
            rr_mult = np.random.uniform(1.3, 2.0)
            temp_add = np.random.uniform(-1.0, 3.0)
            sat_sub = np.random.uniform(5, 15)
            sys_mult = np.random.uniform(0.7, 1.4)
        elif triage_level == 1:  # Yellow (Urgent)
            hr_mult = np.random.uniform(1.0, 1.3)
            rr_mult = np.random.uniform(1.0, 1.4)
            temp_add = np.random.uniform(-0.5, 2.0)
            sat_sub = np.random.uniform(0, 8)
            sys_mult = np.random.uniform(0.85, 1.25)
        else:  # Green (Non-urgent)
            hr_mult = np.random.uniform(0.8, 1.1)
            rr_mult = np.random.uniform(0.9, 1.1)
            temp_add = np.random.uniform(-0.3, 1.0)
            sat_sub = np.random.uniform(0, 3)
            sys_mult = np.random.uniform(0.9, 1.1)
        
        return {
            'heart_rate': max(40, min(200, int(base_hr * hr_mult))),
            'respiratory_rate': max(8, min(40, int(base_rr * rr_mult))),
            'temperature': round(max(35.0, min(42.0, base_temp + temp_add)), 1),
            'oxygen_saturation': max(70, min(100, int(base_sat - sat_sub))),
            'systolic': max(60, min(250, int(base_sys * sys_mult))),
            'diastolic': max(40, min(150, int(base_dia * sys_mult * 0.7)))
        }
    
    def _generate_symptoms(self, triage_level):
        """Generate symptoms based on triage severity."""
        symptoms = {}
        
        if triage_level == 2:  # Critical
            symptom_pool = ['severe chest pain', 'difficulty breathing', 'altered consciousness', 
                          'severe bleeding', 'cardiac arrest', 'stroke symptoms']
        elif triage_level == 1:  # Urgent
            symptom_pool = ['moderate pain', 'fever', 'vomiting', 'dizziness', 
                          'moderate bleeding', 'infection signs']
        else:  # Non-urgent
            symptom_pool = ['mild pain', 'minor cuts', 'cold symptoms', 'routine check',
                          'medication refill', 'minor rash']
        
        # Generate 1-3 symptoms
        num_symptoms = np.random.randint(1, 4)
        selected_symptoms = np.random.choice(symptom_pool, min(num_symptoms, len(symptom_pool)), replace=False)
        
        symptoms['general_01'] = ', '.join(selected_symptoms[:2]) if len(selected_symptoms) > 0 else ''
        symptoms['general_02'] = ', '.join(selected_symptoms[2:]) if len(selected_symptoms) > 2 else ''
        
        # Specific symptoms
        if 'chest pain' in str(selected_symptoms):
            symptoms['chest_pain'] = 'göğüs ağrısı'
        if 'abdominal' in str(selected_symptoms) or 'stomach' in str(selected_symptoms):
            symptoms['abdominal_pain'] = 'karın ağrısı'
        
        return symptoms
    
    def _generate_comorbidities(self, age):
        """Generate comorbidities based on age."""
        if age < 30:
            prob = 0.1
        elif age < 50:
            prob = 0.3
        elif age < 70:
            prob = 0.6
        else:
            prob = 0.8
        
        if np.random.random() < prob:
            num_conditions = np.random.randint(1, 4)
            conditions = np.random.choice(self.comorbidities, num_conditions, replace=False)
            return ', '.join(conditions)
        
        return ''
    
    def _generate_trauma(self, triage_level):
        """Generate trauma data."""
        trauma_data = {}
        
        # 20% chance of trauma
        if np.random.random() < 0.2:
            available_trauma_types = ['travma_ayak', 'travma_bacak', 'travma_batin', 'travma_boyun',
                                    'travma_el', 'travma_gogus', 'travma_goz', 'travma_kafa',
                                    'travma_kol', 'travma_parmak']
            
            if triage_level == 2:  # Severe trauma
                trauma_types = np.random.choice(available_trauma_types, np.random.randint(1, 3), replace=False)
            else:  # Minor trauma
                trauma_types = np.random.choice(available_trauma_types, 1, replace=False)
            
            for trauma_type in trauma_types:
                trauma_data[trauma_type] = f"{trauma_type.replace('travma_', '')} injury"
        
        return trauma_data
    
    def _generate_specialties(self, symptoms, trauma_data):
        """Generate medical specialty assignments."""
        specialties = {}
        
        # Simple rule-based specialty assignment
        if 'chest pain' in str(symptoms) or 'cardiac' in str(symptoms):
            specialties['kardiyoloji'] = 'cardiac evaluation'
        
        if 'breathing' in str(symptoms) or 'respiratory' in str(symptoms):
            specialties['göğüs hastalıkları'] = 'pulmonary assessment'
        
        if trauma_data:
            specialties['genel cerrahi hastalıklar'] = 'trauma surgery'
        
        if 'neurological' in str(symptoms) or 'travma_kafa' in trauma_data:
            specialties['nörolojik hastalıklar'] = 'neurological evaluation'
        
        return specialties
    
    def _generate_timestamp(self):
        """Generate realistic timestamp."""
        # Generate timestamps over the past year
        start_date = datetime.now() - timedelta(days=365)
        random_days = np.random.randint(0, 365)
        random_hours = np.random.randint(0, 24)
        random_minutes = np.random.randint(0, 60)
        
        timestamp = start_date + timedelta(days=random_days, hours=random_hours, minutes=random_minutes)
        return timestamp.strftime('%Y-%m-%d %H:%M:%S')
    
    def _map_triage_to_area(self, triage_level):
        """Map triage level to area name."""
        mapping = {0: 'Yeşil Alan', 1: 'Sarı Alan', 2: 'Kırmızı Alan'}
        return mapping[triage_level]

def generate_rare_cases(num_cases=100):
    """Generate rare and edge cases for stress testing."""
    generator = MedicalTriageDataGenerator()
    
    print(f"Generating {num_cases} rare case scenarios...")
    
    # Force specific rare combinations
    rare_cases = []
    
    # Pediatric cardiac emergencies
    for _ in range(num_cases // 4):
        case = generator._generate_vitals(age=8, triage_level=2)
        case.update({
            'age': np.random.randint(1, 12),
            'symptoms': 'pediatric cardiac arrest, cyanosis',
            'triage': 'Kırmızı Alan'
        })
        rare_cases.append(case)
    
    # Elderly multi-organ failure
    for _ in range(num_cases // 4):
        case = generator._generate_vitals(age=85, triage_level=2)
        case.update({
            'age': np.random.randint(80, 95),
            'symptoms': 'multi-organ failure, sepsis, altered mental status',
            'triage': 'Kırmızı Alan'
        })
        rare_cases.append(case)
    
    # Complex trauma cases
    for _ in range(num_cases // 4):
        case = generator._generate_vitals(age=35, triage_level=2)
        case.update({
            'age': np.random.randint(20, 50),
            'symptoms': 'polytrauma, internal bleeding, head injury',
            'triage': 'Kırmızı Alan'
        })
        rare_cases.append(case)
    
    # Rare disease presentations
    for _ in range(num_cases // 4):
        case = generator._generate_vitals(age=45, triage_level=1)
        case.update({
            'age': np.random.randint(25, 65),
            'symptoms': 'rare autoimmune condition, atypical presentation',
            'triage': 'Sarı Alan'
        })
        rare_cases.append(case)
    
    print(f"✅ Generated {len(rare_cases)} rare cases for stress testing")
    return rare_cases

if __name__ == "__main__":
    # Generate synthetic dataset
    generator = MedicalTriageDataGenerator()
    
    # Create directories if they don't exist
    os.makedirs('data', exist_ok=True)
    
    # Generate main dataset
    df = generator.generate_synthetic_dataset(num_samples=2000, save_path='triaj_data.csv')
    
    # Generate rare cases
    rare_cases = generate_rare_cases(num_cases=200)
    
    print("\n🎉 Synthetic data generation complete!")
    print("📁 Files created:")
    print("   - triaj_data.csv (main dataset)")
    print("   - data/ directory structure")