from ultralytics import YOLO
import cv2
import easyocr
import re
import string
import json
from fuzzywuzzy import fuzz

data = json.load(open("data.json"))

def process_pattern(input_string):
    # Define the regular expression pattern
    pattern = re.compile(r'([a-zA-Z]{1,3})\s*(\d{2,4})\s*([a-zA-Z]{2,3})')

    # Use the pattern to match and transform the input string
    result = re.sub(pattern, r'\1 \2 \3', input_string)

    return result

def remove_punctuation(input_string):
    # Create a translation table mapping each punctuation character to None
    translator = str.maketrans("", "", string.punctuation)

    # Use translate to remove punctuation
    no_punct = input_string.translate(translator)

    return no_punct

def remove_pattern_strings(text):
    pattern = r'\b\d{2}\.\d{2}\b|\b<\d><\d>\.<\d><\d>\b|\b\d{2}\:\d{2}\b'
    # The pattern matches "dd.dd" or "<d><d>.<d><d>" where d is a digit

    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text

def crop_image(image, xyxy):
    return image[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]

def get_xyxys(results):
    xyxys = []
    for result in results.boxes.xyxy.cpu().numpy():
        temp = [int(result[0]), int(result[1]), int(result[2]), int(result[3])]
        xyxys.append(temp)
    return xyxys

def typo_correction(text):
    text_to_correct = text.split(" ")[0]
    maximum = 0
    text_result = ""
    daerah = []
    for d in data["plat_nomor"]:
        # print(fuzz.ratio(text_to_correct, d["kode"]), d["kode"], text_to_correct)
        ratio = fuzz.ratio(text_to_correct, d["kode"])
        if ratio > maximum and ratio > 40:
            maximum = ratio
            text_result = d["kode"]
            daerah = d["daerah"]

    if maximum > 0:
        text = text.replace(text_to_correct, text_result)
        return text, daerah
    return text, None

yolo = YOLO("yolov8s.pt")
plate = YOLO("best.pt")
ocr = easyocr.Reader(['en'], gpu=True)

sample_path = "sample/6.jpg"

img = cv2.imread(sample_path)

#only detect car, truck, bus, motorcycle
results = yolo(img, classes=[2, 5, 7, 8], agnostic_nms=True, conf=0.5)

img_out = results[0].plot()

vehicle_xyxy = get_xyxys(results[0])
vehicle_list = [crop_image(img.copy(), xyxy) for xyxy in vehicle_xyxy]

for idx,vl in enumerate(vehicle_list):
    results_plate = plate(vl, agnostic_nms=True, conf=0.5)
    plate_xyxy = get_xyxys(results_plate[0])
    if len(plate_xyxy) == 0:
        continue
    cv2.rectangle(img_out, (plate_xyxy[0][0]+vehicle_xyxy[idx][0], plate_xyxy[0][1]+vehicle_xyxy[idx][1]), (plate_xyxy[0][2]+vehicle_xyxy[idx][0], plate_xyxy[0][3]+vehicle_xyxy[idx][1]), (0, 255, 0), 2)
    if plate_xyxy[0][2] - plate_xyxy[0][0] < 90:
        continue
    plate_img = crop_image(vl.copy(), plate_xyxy[0])
    plate_img = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
    plate_text = ocr.readtext(plate_img)
    text = ""
    for pt in plate_text:
        text += " " + pt[1].upper().replace(":",".").replace(" ","")
    text = remove_pattern_strings(text)
    text = remove_punctuation(text)
    text = process_pattern(text).strip()
    text, daerah = typo_correction(text)
    cv2.putText(img_out, text, (plate_xyxy[0][0]+vehicle_xyxy[idx][0], plate_xyxy[0][1]+vehicle_xyxy[idx][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(img_out, ','.join(daerah), (plate_xyxy[0][0]+vehicle_xyxy[idx][0], plate_xyxy[0][1]+vehicle_xyxy[idx][1]+30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
cv2.imwrite(f"out/{sample_path.split('/')[-1]}", img_out)