import argparse
import requests
import os

# Simple config mapping for demo images
IMAGE_PATHS = {
    "infected": "Train/melanoma/ISIC_0000139.jpg",
    "not_infected": "Test/no-melanoma/melanoma_9997.jpg"
}

def predict(model_name, image_path):
    api_url = f"http://0.0.0.0:8082/predict/{model_name}"

    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' not found.")
        return

    with open(image_path, "rb") as img_file:
        files = {"file": (os.path.basename(image_path), img_file, "image/jpeg")}
        response = requests.post(api_url, files=files)

    print("Status Code:", response.status_code)
    try:
        print("Response JSON:", response.json())
    except Exception as e:
        print("Error parsing JSON:", e)
        print("Raw Response Text:", response.text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Skin Cancer Detection API Tester")
    parser.add_argument("--model", type=str, required=True, help="Model name (cnn / mobilenetv2)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--infected", action="store_true", help="Test with an infected (melanoma) image")
    group.add_argument("--not-infected", action="store_true", help="Test with a not infected (benign) image")

    args = parser.parse_args()

    image_key = "infected" if args.infected else "not_infected"
    image_path = IMAGE_PATHS[image_key]

    predict(args.model, image_path)


# python test_api.py --model mobilenetv2 --infected


# python test_api.py --model cnn --not-infected


