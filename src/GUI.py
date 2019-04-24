# import the library
from appJar import gui

# handle button events
def press(button):
    if button == "Exit":
        app.stop()
    else:
        print()
        #usr = app.getEntry("Username")
        #pwd = app.getEntry("Password")
        #print("User:", usr, "Pass:", pwd)

# create a GUI variable called app
app = gui("CEC448 EEG APP", "400x200")
app.setBg("orange")
app.setFont(18)

# add & configure widgets - widgets get a name, to help referencing them later
app.addLabel("title", "EEG ML Processor and Classifier")
app.setLabelBg("title", "blue")
app.setLabelFg("title", "orange")

app.addLabelEntry("Files Path")
#app.addLabelSecretEntry("Other Box")

app.setEntryDefault("Files Path", "../desktop/448/")

# link the buttons to the function called press
app.addButtons(["Process", "Exit"], press)

app.setFocus("Files Path")

# start the GUI
app.go()