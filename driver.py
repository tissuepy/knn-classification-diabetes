import math

class Diabetes:

    def __init__(self, gender, age, hypertension, heart_disease, smoking_history,
                     bmi, hba1c_level, blood_glucose_level, diabetes):
            self.gender = gender
            self.age = float(age)  # Convert to float instead of int
            self.hypertension = int(hypertension)  # Keep as int since it's categorical (0 or 1)
            self.heart_disease = int(heart_disease)  # Keep as int
            self.smoking_history = smoking_history  # String, no conversion needed
            self.bmi = float(bmi)  # Convert to float
            self.hba1c_level = float(hba1c_level)  # Convert to float
            self.blood_glucose_level = float(blood_glucose_level)  # Convert to float

            if diabetes is not None:
                self.diabetes = int(diabetes)
            else:
                self.diabetes = None


    def euclidean_distance(self, other):
        return math.sqrt(
            (self.age - other.age) ** 2 +
            (self.hypertension - other.hypertension) ** 2 +
            (self.heart_disease - other.heart_disease) ** 2 +
            (self.bmi - other.bmi) ** 2 +
            (self.hba1c_level - other.hba1c_level) ** 2 +
            (self.blood_glucose_level - other.blood_glucose_level) ** 2
            # Note: 'diabetes' is typically the target variable and shouldn't be included in distance calculation
        )

    def __str__(self):
        return f"{self.age} ({self.bmi})"


