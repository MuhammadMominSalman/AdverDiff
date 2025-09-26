import os
import csv
import torch
import clip
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from collections import Counter
import pandas as pd

# COFIG VAR
image_folder = "/mnt/Data/musa7216/10k/bdd100k/images/10k/train/"
output_csv = "clip_image_classification.csv"
device = "cuda" if torch.cuda.is_available() else "cpu"
df_folder = "/mnt/Data/musa7216/10k/labels/"

df = pd.read_csv(os.path.join(df_folder, "train_df.csv"))
df = df[["name", "timeofday", "weather"]]
df= df[df["weather"] != "undefined"]
df= df[df["weather"] != "foggy"]
df= df[df["timeofday"] != "undefined"]
# df = df.drop_duplicates(subset=['name'], keep='first')
df["timeofday"]= df.apply(lambda x: x['timeofday'].replace('daytime','day'), axis=1)
df["timeofday"]= df.apply(lambda x: x['timeofday'].replace('dawn/dusk','dawn'), axis=1)
df["weather"]= df.apply(lambda x: x['weather'].replace('partly cloudy','clear'), axis=1)
df["weather"]= df.apply(lambda x: x['weather'].replace('overcast','cloudy'), axis=1)


weather_classes = ["clear", "snowy", "rainy", "cloudy"]
timeofday_classes = ["dawn", "night", "day"]

model, preprocess = clip.load("ViT-L/14", device=device)

def classify_image(image, prompts):
    text = clip.tokenize(prompts).to(device)
    image = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        predicted_index = similarity.argmax().item()

    return prompts[predicted_index]


results = []
weather_counts = Counter()
timeofday_counts = Counter()

image_files = [f for f in os.listdir(image_folder) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]

df=df[df["name"].isin(image_files)]

correct = 0
correct_w = 0
total = len(image_files)
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    filename = df["name"][index]
    w_true = df["weather"][index]
    tod_true = df["timeofday"][index]
    try:
        image_path = os.path.join(image_folder, filename)
        image = Image.open(image_path).convert("RGB")

        weather = classify_image(image, weather_classes)
        timeofday = classify_image(image, timeofday_classes)

        results.append([filename, weather, timeofday])
        weather_counts[weather] += 1
        timeofday_counts[timeofday] += 1
        if weather == w_true:
            correct_w +=1
        if timeofday == tod_true:
            correct +=1
    except Exception as e:
        print(f"Error processing {filename}: {e}")

acc_tod = correct/total
acc_w = correct_w/total
print("Weather Accurcay:", acc_w)
print("Time of Day Accuracy:", acc_tod)



with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "weather", "timeofday"])
    writer.writerows(results)

print(f"\nResults saved to {output_csv}")


plt.figure(figsize=(12, 5))


# WEATHER PLOT
plt.subplot(1, 2, 1)
plt.bar(weather_counts.keys(), weather_counts.values(), color="skyblue")
plt.title("Weather Distribution")
plt.xlabel("Weather")
plt.ylabel("Number of Images")

# TOD PLOT
plt.subplot(1, 2, 2)
plt.bar(timeofday_counts.keys(), timeofday_counts.values(), color="orange")
plt.title("Time of Day Distribution")
plt.xlabel("Time of Day")
plt.ylabel("Number of Images")

plt.tight_layout()
plt.savefig("classification_distribution.png")
plt.show()
