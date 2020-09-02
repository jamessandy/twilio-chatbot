**Analog Image Caption Package**

When you're done, you can delete the content in this README and update the file with details for others getting started with your repository.



---

## How to try this package ?


This package is very easy to use and has 2  fuctions which are:

1. extract_features() this function simply take the image and extract the features from it.
2. generate_desc() this function now generates a description for the Image.

---

## Demo in a flask app
@app.route("/generateCaption", methods=["POST"])
def generateCaption():
    image = request.files['image']
    img = image.read()

    # convert string of image data to uint8
    nparr = np.fromstring(img, np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))
   

    photo = extract_features(img)
    # generate description
    caption = generate_desc(model, tokenizer, photo, max_length)

   
    return render_template("results.html", image=image, caption=caption)

---

## Demo in a python script
photo = extract_features(img_path)
img = Image.open(img_path)
description = generate_desc(model, tokenizer, photo, max_length)
print("\n\n")
print(description)
plt.imshow(img)


Hope this helps you understand the package and use it in a project.
