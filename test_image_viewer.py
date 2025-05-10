import cv2
#test
# Specificeer het pad naar het PPM-bestand
image_path = r'C:\Users\Rachid\Documents\GTSRB_Dataset\GTSRB_Final_Training_Images\GTSRB\Final_Training\Images\00000\00005_00017.ppm'

# Lees de afbeelding in
img = cv2.imread(image_path)

# Toon de afbeelding in een nieuw venster
cv2.imshow('Verkeersbord', img)

# Wacht op een toetsdruk om het venster te sluiten
cv2.waitKey(0)
cv2.destroyAllWindows()