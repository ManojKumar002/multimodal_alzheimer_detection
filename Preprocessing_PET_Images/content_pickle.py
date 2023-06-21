import pickle

objects = []
with (open("mri_meta.pkl", "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break
print(type(objects))
