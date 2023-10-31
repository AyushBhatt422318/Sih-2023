from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import logging
from flask_cors import CORS
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
from io import BytesIO

app = Flask(__name__)
CORS(app)

# Define the full path to your saved model
MODEL_PATH = "C:/Users/Acer/sih all codes/batch_32_001.h5"  # Update with the actual path

# Load the saved model
try:
    loaded_model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    logging.error(f"Error loading the model: {str(e)}")

# Define a function to preprocess an image
# Define a function to preprocess an image
def preprocess_image(image):
    try:
        desired_size = (180, 180)  # Define the desired size

        # Save the uploaded image to a temporary file
        temp_file_path = "C:/Users/Acer/sih all codes/temp_image.jpg"
        image.save(temp_file_path)

        # Load the saved image
        img = load_img(temp_file_path, target_size=desired_size)
        if img is None:
            logging.error("Image loading failed.")
            return None

        img_array = img_to_array(img)
        if img_array is None:
            logging.error("Image to array conversion failed.")
            return None

        img_array = img_array / 255.0  # Normalize pixel values

        # Remove the temporary file
        os.remove(temp_file_path)

        return img_array
    except Exception as e:
        logging.error(f"Error preprocessing image: {str(e)}")
        return None


# Define an API endpoint for disease detection
@app.route('/detect_disease', methods=['POST'])
def detect_disease():
    try:
        # Get the uploaded image from the request
        uploaded_image = request.files['image']

        if not uploaded_image:
            return jsonify({'error': 'No image file provided'})

        # Preprocess the image
        preprocessed_image = preprocess_image(uploaded_image)

        if preprocessed_image is None:
            return jsonify({'error': 'Error in image preprocessing'})

        # Make predictions using the loaded model
        predictions = loaded_model.predict(np.array([preprocessed_image]))

        # Interpret the predictions based on your project's disease class names
        # Replace class_names with your list of disease class names
        class_names = ['Acne', 'Actinic keratosis', 'Eczema herpeticum', 'Herpes', 'Measles', 'Rosacea', 'Tinea corporis', 'Tinea pedis', 'Viral warts', 'Vitiligo']
        class_index = np.argmax(predictions)
        diseaselabel = class_names[class_index]

        # Get the confidence score
        confidence = predictions[0][class_index]
        
        description = ''
        localnames = ''

        # Return the results as JSON
        result = {
            'diseaselabel': diseaselabel,
            'confidence': float(confidence) if confidence is not None else 0
        }
        
        if diseaselabel == "Acne":
                result['description'] ='''A skin condition that occurs when hair follicles plug with oil and dead skin cells.
Acne is most common in teenagers and young adults.
Symptoms range from uninflamed blackheads to pus-filled pimples or large, red and tender bumps.
Treatments include over-the-counter creams and cleanser, as well as prescription antibiotics.'''
                result['localnames']='''Hindi: मुँहासे (Munhase)
Tamil: முகப்பரு (Mukapparu)
Bengali: মুখরোগ (Mukhrog)
Telugu: ముఖ రోగం (Mukha Rogam)
Kannada: ಮುಖ ಕುರುಪು (Mukha Kurupu)
Marathi: मुखाच्या गोडामार्या (Mukhachya Godamarya)'''
        
        elif diseaselabel == "Actinic keratosis":
                result['description'] ='''A rough, scaly patch on the skin caused by years of sun exposure.
Actinic keratoses usually affects older adults. Reducing sun exposure can help reduce risk.
It is most common on the face, lips, ears, back of hands, forearms, scalp and neck. The rough, scaly skin patch enlarges slowly and usually causes no other signs or symptoms. A lesion may take years to develop.
Because it can become cancerous, it's usually removed as a precaution.'''
                result['localnames']='''Hindi: Radke or Dhoop ki Bimari (Sunlight Disease)
Tamil: Paruvi Chivithi or Udaivichu Chivithi (Lesions due to Sun Exposure)
Bengali: Alokoprakash Promoksho (Light-Induced Lesions)
Telugu: Prakasha Vyadhi (Disease caused by Light)
Kannada: Belakina Asuye (Lesions due to Sunlight)
Marathi: Alokaprakash Rog (Light-Related Disease)'''
        
        elif diseaselabel == "Eczema herpeticum":
                result['description'] ='''Eczema herpeticum, also known as a form of Kaposi varicelliform eruption caused by viral infection, usually with the herpes simplex virus (HSV), is an extensive cutaneous vesicular eruption that arises from pre-existing skin disease, usually atopic dermatitis (AD). Children with AD have a higher risk of developing eczema herpeticum, in which HSV type 1 (HSV-1) is the most common pathogen.

Eczema herpeticum can be severe, progressing to disseminated infection and death if untreated.1 Bacterial superinfection and bacteremia are usually the complications that cause mortality. We present a case in which eczema herpeticum was misdiagnosed as impetigo during a patient’s initial treatment. Detailed history taking and characteristic cutaneous findings can help clinicians make an accurate diagnosis.'''
                result['localnames']='''will add this later'''
                
        elif diseaselabel == "Herpes":
            result['description'] ='''Herpes simplex virus (HSV), known as herpes, is a common infection that can cause painful blisters or ulcers. It primarily spreads by skin-to-skin contact. It is treatable but not curable.

There are two types of herpes simplex virus.

Type 1 (HSV-1) mostly spreads by oral contact and causes infections in or around the mouth (oral herpes or cold sores). It can also cause genital herpes. Most adults are infected with HSV-1.

Type 2 (HSV-2) spreads by sexual contact and causes genital herpes.

Most people have no symptoms or only mild symptoms. The infection can cause painful blisters or ulcers that can recur over time. Medicines can reduce symptoms but can’t cure the infection.

Recurrent symptoms of both oral and genital herpes may be distressing. Genital herpes may also be stigmatizing and have an impact on sexual relationships. However, in time, most people with either kind of herpes adjust to living with the infection.'''
            result['localnames']='''Hindi: अल्सर (Alsar)
Tamil: புழுக்கள் (Puzhukkal)
Bengali: সর্পদান্তু (Sarpadantu)
Telugu: విషవ్యాధి (Vishavyadhi)
Kannada: ಹೆರ್ಪೀಸ್ (Herpīs)
Marathi: सर्पाट (Sarpaṭ)'''
        
        elif diseaselabel == "Measles":
            result['description'] ='''A viral infection that's serious for small children but is easily preventable by a vaccine.
The disease spreads through the air by respiratory droplets produced from coughing or sneezing.
Measles symptoms don't appear until 10 to 14 days after exposure. They include cough, runny nose, inflamed eyes, sore throat, fever and a red, blotchy skin rash.
There's no treatment to get rid of an established measles infection, but over-the-counter fever reducers or vitamin A may help with symptoms.'''
            result['localnames']='''Hindi: खसरा (Khasra)
Tamil: ரோகம் (Rogam)
Bengali: হাস্পশ (Haspash)
Telugu: రొగం (Rogam)
Kannada: ಮೀಸಲ್ಸ್ (Mīsals)
Marathi: सुसला (Susla)'''
                
        elif diseaselabel == "Rosacea":
            result['description'] ='''A condition that causes redness and often small, red, pus-filled bumps on the face.
Rosacea most commonly affects middle-aged women with fair skin. It can be mistaken for acne or other skin conditions.
Key symptoms are facial redness with swollen red bumps and small visible blood vessels.
Treatments such as antibiotics or anti-acne medication can control and reduce symptoms. Left untreated, it tends to worsen over time.'''
            result['localnames']='''Hindi: फुलवाड़ी (Phulwadi)
Tamil: முக இழுக்கம் (Mukha Izhukkam)
Bengali: মুখের রোজেসিয়া (Mukher Rojeshia)
Telugu: ముఖంతన ప్రదర్శన (Mukhanta Pradarshana)
Kannada: ಮುಖದ ಮುಕ್ಕಣಿಗೆ (Mukhada Mukkaṇige)
Marathi: मुखाच्या अकडण्या (Mukhachya Akadanya)'''
        
        elif diseaselabel == "Tinea corporis":
            result['description'] ='''Tinea corporis is a superficial fungal skin infection of the body caused by dermatophytes. Tinea corporis can be found worldwide. It is specifically defined by the location of the lesions that may involve the trunk, neck, arms, and legs. Alternative names are used for dermatophyte infections that affect the other areas of the body. These include the scalp (tinea capitis), the face (tinea faciei), hands (tinea manuum), the groin (tinea cruris), and feet (tinea pedis). This activity highlights the evaluation, diagnosis, treatment, and complications of tinea corporis.'''
            result['localnames']='''Hindi: दाद (Daad)
Tamil: தாது (Thadu)
Bengali: দাদ (Dad)
Telugu: దాడి (Daadi)
Kannada: ದಾಡಿ (Daadi)
Marathi: दाद (Daad)'''
            
        elif diseaselabel == "Tinea pedis":
            result['description'] ='''A fungal infection that usually begins between the toes.
Athlete's foot commonly occurs in people whose feet have become very sweaty while confined within tight-fitting shoes.
Symptoms include a scaly rash that usually causes itching, stinging and burning. People with athlete's foot can have moist, raw skin between their toes.
Treatment involves topical anti-fungal medication.'''
            result['localnames']='''Hindi: पैरों का फफूणा (Pairo ka Phaphoona)
Tamil: கால் புழுக்கம் (Kaalu Puzhukkam)
Bengali: পায়ের দাদ (Payer Dad)
Telugu: పాదానాను (Paadananu)
Kannada: ಕಾಲು ದಾಡಿ (Kaalu Daadi)
Marathi: पायांच्या अकडण्या (Paayaanchya Akadanya)'''
            
        
        elif diseaselabel == "Viral warts":
            result['description'] ='''Warts are a type of skin infection caused by the human papillomavirus (HPV). The infection causes rough, skin-colored bumps to form on the skin. The virus is contagious. You can get warts from touching someone who has them. Warts most commonly appear on the hands, but they can also affect the feet, face, genitals and knees.'''
            result['localnames']='''Hindi: वायरल वार्ट्स (Viral Warts)
Tamil: வைரஸ் வர்ட்ஸ் (Vairas Varṭs)
Bengali: ভাইরাল ওয়ার্টস (Viral Warts)
Telugu: వైరల్ వార్ట్స్ (Vairal Warts)
Kannada: ವೈರಲ್ ವಾರ್ಟ್ಸ್ (Vairal Varts)
Marathi: व्हॉयरल वॉर्ट्स (Viral Warts)'''
            
        else:
            result['description'] ='''A disease that causes the loss of skin colour in blotches.
Vitiligo occurs when pigment-producing cells die or stop functioning.
Loss of skin colour can affect any part of the body, including the mouth, hair and eyes. It may be more noticeable in people with darker skin.
Treatment may improve the appearance of the skin but doesn't cure the disease.'''
            result['localnames']='''Hindi: श्वेतकुष्ठ (Shwetakushth)
Tamil: வெள்ளைக் குச்சி (Vellai Kucchi)
Bengali: সাদা ব্যাদি (Shada Byadi)
Telugu: బిళ్ళ పిలక (Billapilaka)
Kannada: ಬಿಳಿಯ ಗುರುತು (Biliya Gurutu)
Marathi: पांढर्या पिटाची बिमारी (Pandharya Pitaachi Bimari)'''
            
                
        
            
        
        return jsonify(result)
    except Exception as e:
        logging.error(f"Error in disease detection endpoint: {str(e)}")
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Disable debug mode for production
    app.run(host='0.0.0.0', port=5000)
