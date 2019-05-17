# import the library
from appJar import gui

import kFold
import sLDA

# Type <filename>.<methodname> to run the method from each file

#sLDA.runSLDA()

# handle button events
def press(button):
    if button == "Exit":
        app.stop()
    elif button == "k-fold":
        kFoldResult = kFold.runKFold(app.getEntry("Files Path"))
        app.setLabel("output", ("kFold accuracy:\n" + str(kFoldResult)) );
        app.reloadImage("AUC", "empty.png")
        #print("wow I got a result it is:" + str(kFoldResult))
        #print("ff", app.getEntry("Files Path"))
        #usr = app.getEntry("Username")
        #pwd = app.getEntry("Password")
        #print("User:", usr, "Pass:", pwd)
    else:
        AUCResult = sLDA.runSLDA(app.getEntry("Files Path"))
        app.setLabel("output", ("kFold accuracy:\n" + str(AUCResult)) );
        app.reloadImage("AUC", "aucFig.png")
        

# create a GUI variable called app
app = gui("CEC448 EEG APP", "800x600")
app.setBg("cyan")
app.setFont(18)

# add & configure widgets - widgets get a name, to help referencing them later
app.addLabel("title", "EEG ML Processor and Classifier")
app.setLabelBg("title", "blue")
app.setLabelFg("title", "cyan")

app.addLabelEntry("Files Path")
app.setEntry("Files Path", "../desktop/448/")
#app.addLabelSecretEntry("Other Box")

#app.setEntryDefault("Files Path", "../desktop/448/")

# link the buttons to the function called press
app.addButtons(["k-fold", "AUC", "Exit"], press)

app.addLabel("output", "")
#app.setLabel("output", "oof")

app.setImageLocation("../desktop/448")
app.startLabelFrame("")
app.addImage("AUC", "empty.png")
app.stopLabelFrame()
#app.setBgImage("testImage11.gif")

app.setFocus("Files Path")

# start the GUI
app.go()