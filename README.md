# intel_cnn

# TO RUN
In order to run this program, you must first install all the necessary libraries. **It is also useful to set up a virtual environment**, there is a good bit to download.
run `pip install -r requirements.txt` to install all libraries. 
# Understanding the output
To understand this output, first look at the server that is being created using flask. This is run as a separate thread to allow automatic updates to the template from the loop where the CNN makes its predictions. This section is not necessary and was created purely for demo purposes. You can comment the line out at the very end of the program that runs the server and this will leave only the CNN command line output. 
`if __name__ == '__main__':
    threading.Thread(target=app.run).start() #comment this line out to run without webserver
    main()`
to run the program itself, `python cnn_audio.py`
