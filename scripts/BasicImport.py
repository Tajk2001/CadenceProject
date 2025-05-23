from fitparse import FitFile

file_path = r"C:\Users\User\Desktop\New folder\165Crank.fit"
fitfile = FitFile(file_path)

# Print all messages and fields in the FIT file
for message in fitfile.get_messages():
    print(f"Message: {message.name}")
    for field in message:
        print(f"  {field.name}: {field.value}")


