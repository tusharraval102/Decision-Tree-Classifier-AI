import tkinter as tk
from tkinter import filedialog, messagebox
from decisionTree import decisionTree, classify, convertInput
import numpy as np
from sklearn import preprocessing
from PIL import Image, ImageTk
import ast

# GUI
class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        # Label for the dataset entry
        self.dataset_label = tk.Label(self, text="Dataset")
        self.dataset_label.grid(row=0, column=0, sticky="w")
        
        # Text field for entering the dataset
        self.dataset_entry = tk.Text(self, height=10)
        self.dataset_entry.grid(row=1, column=0, sticky="nsew")

        # Button for loading the dataset
        self.load_button = tk.Button(self)
        self.load_button["text"] = "Load Data"
        self.load_button["command"] = self.load_data
        self.load_button.grid(row=2, column=0)

        # Label for the input entry
        self.input_label = tk.Label(self, text="Input")
        self.input_label.grid(row=3, column=0, sticky="w")
        # Text field for entering the input
        self.input_entry = tk.Entry(self)
        self.input_entry.grid(row=4, column=0, sticky="nsew")

        # Button for making predictions
        self.predict_button = tk.Button(self)
        self.predict_button["text"] = "Predict"
        self.predict_button["command"] = self.predict
        self.predict_button.grid(row=5, column=0)

        # Label for the output
        self.output_label = tk.Label(self, text="Output")
        self.output_label.grid(row=6, column=0, sticky="w")

        # Text field for displaying the output
        self.output_entry = tk.Entry(self)
        self.output_entry.grid(row=7, column=0, sticky="nsew")
        
        # Configure the row and column weights
        self.master.grid_columnconfigure(0, weight=1)
        for i in range(8):
            self.master.grid_rowconfigure(i, weight=1)


    # Load the dataset and train the model
    def load_data(self):
        # Get the dataset from the text field
        dataset_str = self.dataset_entry.get('1.0', 'end')
        dataset = ast.literal_eval(dataset_str)
        dataset = np.array(dataset)
        
        # Check if all rows have the same number of columns
        num_cols = len(dataset[0])
        for row in dataset:
            if len(row) != num_cols:
                messagebox.showinfo("Error", "All rows must have the same number of columns")
                return

         # Initialize the label encoders and train the decision tree
        self.labelEncoders = [preprocessing.LabelEncoder() for i in range(len(dataset[0]))]
        self.clf = decisionTree(dataset, self.labelEncoders)
        messagebox.showinfo("Info", "Data loaded and model trained")


        # Display the decision tree
        img = Image.open("DecisionTree.png")
        self.tree_image = ImageTk.PhotoImage(img)
        self.tree_label = tk.Label(self.master, image=self.tree_image)
        self.tree_label.pack()

    # Make a prediction based on the input
    def predict(self):
        try:
            for i, le in enumerate(self.labelEncoders[:-1]):
                print(le.classes_)
            # Get the input from the text field
            input_str = self.input_entry.get()
            input_data = ast.literal_eval(input_str)
            input_data = convertInput(input_data, self.labelEncoders)
            output = classify(self.clf, input_data, self.labelEncoders)
            self.output_entry.config(state='normal')
            self.output_entry.delete(0, tk.END)  # Modify this line
            self.output_entry.insert(tk.END, str(output))
            self.output_entry.config(state='disabled')        
            
        except ValueError as e:
            messagebox.showinfo("Error", "Input is not a valid Python literal or container: " + str(e))

root = tk.Tk()
app = Application(master=root)
app.mainloop()