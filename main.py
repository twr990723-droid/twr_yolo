from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageDraw
from matplotlib.pyplot import imshow
import numpy as np
import torch
import matplotlib.pyplot as plt

# Initialize MTCNN for face detection
mtcnn = MTCNN()

# Load pre-trained Inception ResNet model
#resnet = InceptionResnetV1(pretrained='casia-webface').eval()

# Load two face images to be verified
img1 = Image.open('face_images/ruby_lin.jpg').convert("RGB")
img2 = Image.open('face_images/ruby_lin1.jpg').convert("RGB")

print(f"img1.shape={np.asarray(img1).shape}")
#print(f"img2.shape={np.asarray(img2).shape}")

# Detect faces and extract embeddings
faces1, _ = mtcnn.detect(img1)
faces2, _ = mtcnn.detect(img2)

print(f"---Face Detection Result of Image-1---")
if faces1 is not None:
    for box in faces1:
        #print(box)
        # Draw bounding boxes on the image
        draw = ImageDraw.Draw(img1)
        draw.rectangle(box.tolist(), outline='red', width=3)
plt.imshow(img1)
plt.show()
#imshow(np.asarray(img1))

print(f"---Face Detection Result of Image-2---")
if faces2 is not None:
    for box in faces2:
        #print(box) # [x1,y1,x2,y2]
        # Draw bounding boxes on the image
        draw = ImageDraw.Draw(img2)
        draw.rectangle(box.tolist(), outline='red', width=3)
# img1.show()
plt.imshow(img2)
plt.show()
#imshow(np.asarray(img2))

resnet_classify = InceptionResnetV1(pretrained='vggface2', classify=True).eval()
resnet = InceptionResnetV1(pretrained='vggface2').eval()

#resnet = InceptionResnetV1(pretrained='casia-webface').eval()



if faces1 is not None and faces2 is not None:
    aligned1 = mtcnn(img1)
    aligned2 = mtcnn(img2)
    #print(aligned1.shape)
    aligned1 = aligned1.unsqueeze(0)
    aligned2 = aligned2.unsqueeze(0)

    embeddings1 = resnet(aligned1).detach()
    embeddings2 = resnet(aligned2).detach()

    logit1 = resnet_classify(aligned1).detach()
    max_value1, max_index1 = torch.max(logit1, dim=1)

    #print(f"---embeddings1.shape={embeddings1.shape}")
    #print(f"---logit1.shape={logit1.shape}")

    # print(f"---logit1.shape={logit1.shape}")
    # print(f"---max_values1={max_value1}")
    print(f"---max_index1={max_index1}")

    logit2 = resnet_classify(aligned2).detach()
    max_value2, max_index2 = torch.max(logit2, dim=1)

    # print(f"---logit2.shape={logit2.shape}")
    # print(f"---max_values2={max_value2}")
    print(f"---max_index2={max_index2}")

    #logit2 = resnet_classify(aligned2).detach()

    # Calculate the Euclidean distance between embeddings
    distance = (embeddings1 - embeddings2).norm().item()

    print(f"---distance={distance}")
    if distance < 1.2:  # You can adjust the threshold for verification
        print("Same person")
    else:
        print("Different persons")

