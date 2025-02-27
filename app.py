# from flask import Flask, render_template, request, redirect
# from PIL import Image
# import io
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision.transforms as transforms

# # Définition de la même architecture que lors de l'entraînement
# class Net(nn.Module):
#     def __init__(self, num_classes=10):
#         super(Net, self).__init__()
#         # Bloc 1
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
#         self.bn1   = nn.BatchNorm2d(32)
#         self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
#         self.bn2   = nn.BatchNorm2d(32)
#         self.pool1 = nn.MaxPool2d(2, 2)
#         self.dropout1 = nn.Dropout(0.25)
#         # Bloc 2
#         self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.bn3   = nn.BatchNorm2d(64)
#         self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
#         self.bn4   = nn.BatchNorm2d(64)
#         self.pool2 = nn.MaxPool2d(2, 2)
#         self.dropout2 = nn.Dropout(0.25)
#         # Couches entièrement connectées
#         self.flatten = nn.Flatten()
#         self.fc1 = nn.Linear(64 * 8 * 8, 512)  # 32x32 -> pool1:16x16 -> pool2:8x8
#         self.bn5 = nn.BatchNorm1d(512)
#         self.dropout3 = nn.Dropout(0.5)
#         self.fc2 = nn.Linear(512, num_classes)
    
#     def forward(self, x):
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = self.pool1(x)
#         x = self.dropout1(x)
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = F.relu(self.bn4(self.conv4(x)))
#         x = self.pool2(x)
#         x = self.dropout2(x)
#         x = self.flatten(x)
#         x = F.relu(self.bn5(self.fc1(x)))
#         x = self.dropout3(x)
#         x = self.fc2(x)
#         return x

# # Chargement du modèle
# device = torch.device("cpu")  # ou "cuda" si vous utilisez un GPU
# model = Net()
# model.load_state_dict(torch.load('mon_modele.pth', map_location=device))
# model.eval()

# # Transformation de l'image : redimensionnement et conversion en tenseur
# transform = transforms.Compose([
#     transforms.Resize((32, 32)),  # CIFAR-10: 32x32
#     transforms.ToTensor()
# ])

# # Liste des classes
# class_names = ['avion', 'automobile', 'oiseau', 'chat', 'cerf', 
#                'chien', 'grenouille', 'cheval', 'bateau', 'camion']

# app = Flask(__name__)

# @app.route("/", methods=["GET", "POST"])
# def index():
#     prediction = None
#     if request.method == "POST":
#         if 'file' not in request.files:
#             return redirect(request.url)
#         file = request.files['file']
#         if file.filename == "":
#             return redirect(request.url)
#         if file:
#             # Ouvrir l'image et la convertir en RGB
#             image = Image.open(file.stream).convert("RGB")
#             # Appliquer les transformations
#             img_tensor = transform(image)
#             # Ajouter la dimension batch
#             img_tensor = img_tensor.unsqueeze(0)
#             # Prédiction
#             with torch.no_grad():
#                 outputs = model(img_tensor)
#                 probs = F.softmax(outputs, dim=1)
#                 pred_idx = torch.argmax(probs, dim=1).item()
#                 prediction = class_names[pred_idx]
#     return render_template("index.html", prediction=prediction)

# if __name__ == "__main__":
#     app.run(debug=True)
from flask import Flask, render_template, request
from PIL import Image
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

# Définition du modèle (Doit être identique au modèle entraîné)
class Net(nn.Module):
    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4   = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.25)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        x = self.flatten(x)
        x = F.relu(self.bn5(self.fc1(x)))
        x = self.dropout3(x)
        x = self.fc2(x)
        return x

# Charger le modèle
device = torch.device("cpu")
model = Net()
model.load_state_dict(torch.load('mon_modele.pth', map_location=device))
model.eval()

# Transformation des images
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

# Classes du modèle
class_names = ['avion', 'automobile', 'oiseau', 'chat', 'cerf', 
               'chien', 'grenouille', 'cheval', 'bateau', 'camion']

# Dossier pour stocker les images uploadées
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    image_path = None

    if request.method == "POST":
        if 'file' not in request.files:
            return render_template("index.html", prediction="Aucun fichier sélectionné", image_path=None)
        
        file = request.files['file']
        if file.filename == "":
            return render_template("index.html", prediction="Aucun fichier sélectionné", image_path=None)

        if file:
            # Sauvegarde de l'image uploadée
            image_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(image_path)

            # Prétraitement de l'image
            image = Image.open(image_path).convert("RGB")
            img_tensor = transform(image).unsqueeze(0)

            # Prédiction
            with torch.no_grad():
                outputs = model(img_tensor)
                probs = F.softmax(outputs, dim=1)
                pred_idx = torch.argmax(probs, dim=1).item()
                prediction = class_names[pred_idx]

    return render_template("index.html", prediction=prediction, image_path=image_path)

if __name__ == "__main__":
    app.run(debug=True)

