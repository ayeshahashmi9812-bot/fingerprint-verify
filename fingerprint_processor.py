import cv2
import numpy as np
from PIL import Image
import io
import os

class FingerprintVerifier:
    def __init__(self):
        self.registered_embeddings = {}
        print(" Fingerprint Verifier Initialized")
    
    def preprocess_image(self, image_bytes):
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image_np = np.array(image)
            
            if len(image_np.shape) == 3:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            
            image_np = cv2.resize(image_np, (200, 200))
            
            image_np = cv2.equalizeHist(image_np)
            
            return image_np
        except Exception as e:
            print(f"Preprocessing error: {e}")
            return None
    
    def extract_features(self, image_bytes):
        try:
            processed_image = self.preprocess_image(image_bytes)
            if processed_image is None:
                return None
            
            texture_features = self.extract_texture_features(processed_image)
            frequency_features = self.extract_frequency_features(processed_image)
            statistical_features = self.extract_statistical_features(processed_image)
            
            combined_features = np.concatenate([
                texture_features, 
                frequency_features, 
                statistical_features
            ])
            
            return combined_features
            
        except Exception as e:
            print(f" Feature extraction error: {e}")
            return None
    
    def extract_texture_features(self, image):
        features = []
        kernels = []
        
        for theta in np.arange(0, np.pi, np.pi/4):
            kernel = cv2.getGaborKernel((21, 21), 5.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
            kernels.append(kernel)
        
        for kernel in kernels:
            filtered = cv2.filter2D(image, cv2.CV_8UC3, kernel)
            features.extend([np.mean(filtered), np.std(filtered)])
        return np.array(features)
    
    def extract_frequency_features(self, image):
        fft = np.fft.fft2(image)
        fft_shift = np.fft.fftshift(fft)
        magnitude_spectrum = np.log(np.abs(fft_shift) + 1)
        
        features = []
        h, w = magnitude_spectrum.shape
        quarters = [
            magnitude_spectrum[:h//2, :w//2],
            magnitude_spectrum[:h//2, w//2:],
            magnitude_spectrum[h//2:, :w//2],
            magnitude_spectrum[h//2:, w//2:]
        ]
        
        for quarter in quarters:
            features.extend([np.mean(quarter), np.std(quarter)])
        
        return np.array(features)
    
    def extract_statistical_features(self, image):
        features = [
            np.mean(image),
            np.std(image),
            np.median(image),
            np.min(image),
            np.max(image),
            np.percentile(image, 25),
            np.percentile(image, 75),
        ]
        return np.array(features)
    
    def register_fingerprint(self, person_id, image_bytes):
        features = self.extract_features(image_bytes)
        if features is not None:
            self.registered_embeddings[person_id] = features
            print(f"Registered: {person_id}")
            return True
        return False
    
    def load_dataset(self, dataset_path="dataset"):
        
        try:
            print(" Loading dataset...")
            
            if not os.path.exists(dataset_path):
                print(f" Dataset folder not found: {dataset_path}")
                return False
            
            registered_count = 0
        
            for person_folder in os.listdir(dataset_path):
                person_path = os.path.join(dataset_path, person_folder)
                
                if os.path.isdir(person_path):
                    print(f"Processing: {person_folder}")
                    
                    for image_file in os.listdir(person_path):
                        if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            image_path = os.path.join(person_path, image_file)
                            
                            try:
                                with open(image_path, 'rb') as f:
                                    image_bytes = f.read()
                                
                                if self.register_fingerprint(person_folder, image_bytes):
                                    registered_count += 1
                                    print(f"Registered: {image_file}")
                                    break  # Sirf 1 image register karo har person ki
                                else:
                                    print(f" Failed: {image_file}")
                                    
                            except Exception as e:
                                print(f" Error with {image_file}: {e}")
                            break  
            
            print(f" Dataset loaded! Total registered: {registered_count}")
            return True
            
        except Exception as e:
            print(f" Dataset loading error: {e}")
            return False
    
    def verify_fingerprint(self, image_bytes, threshold=0.90):

        input_features = self.extract_features(image_bytes)
        if input_features is None:
           return {"matched": False, "confidence": 0.0}

        if not self.registered_embeddings:
           return {"matched": False, "confidence": 0.0}

        best_score = -999
        best_similarity = 0
        best_l1 = 0
        best_match = None

        for person_id, stored_features in self.registered_embeddings.items():

            min_len = min(len(input_features), len(stored_features))
            input_vec = input_features[:min_len]
            stored_vec = stored_features[:min_len]
            
            input_vec = input_vec / (np.linalg.norm(input_vec) + 1e-8)
            stored_vec = stored_vec / (np.linalg.norm(stored_vec) + 1e-8)

            similarity = np.dot(input_vec, stored_vec)

            l1_distance = np.mean(np.abs(input_vec - stored_vec))

            score = similarity - l1_distance

            if score > best_score:
                best_score = score
                best_similarity = similarity
                best_l1 = l1_distance
                best_match = person_id

        matched = (best_similarity >= threshold) and (best_l1 < 0.12)

        return {
            "matched": bool(matched),
            "confidence": round(float(best_similarity), 2),
            "distance": float(best_l1),
            "person_id": best_match if matched else None
        }

    def get_registered_count(self):
        return len(self.registered_embeddings)

fingerprint_verifier = FingerprintVerifier()
