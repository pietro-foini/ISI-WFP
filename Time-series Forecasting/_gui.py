import tkinter as tk
import numpy as np

class gui:
    def __init__(self):
        # Create window.
        self.root = tk.Tk()
        # Create frame.
        self.frame = tk.Frame(self.root)
        self.frame.grid()
        # Create canvas.
        self.canvas = tk.Canvas(self.frame, width = 750)
        
        # Define output.
        self.output = None
    
    # Define widget.
    def Label(self, root, text, rowno, colno, size, **kwargs):
        lb = tk.Label(root, text = text, **kwargs)
        lb.grid(row = rowno, column = colno)
        lb.config(font = ("helvetica", size))
    
    # Define widget.
    def EntryText(self, root, textVariable, rowno, colno):
        entry = tk.Entry(root, textvariable = textVariable)
        entry.grid(row = rowno, column = colno)
    
    # Define widget.
    def Checkbutton(self, root, text, variable, rowno, colno, disable = False):
        chkbtn = tk.Checkbutton(root, text = text, variable = variable)
        chkbtn.grid(row = rowno, column = colno)
        if disable:
            chkbtn.configure(state = tk.DISABLED)
    
    # Define widget.
    def Listbox(self, root, name, variableValues, rowno, colno, defaultValues):
        # Set label listbox.
        self.Label(root, name, rowno, colno, size = 10, wraplength = 100)
        # Set listbox.
        lstbox = tk.Listbox(root, listvariable = variableValues, selectmode = tk.MULTIPLE, exportselection = 0)
        lstbox.grid(row = rowno+1, column = colno, columnspan = 1)
        # Set default selection.
        if not defaultValues[1] is None:
            if name in defaultValues[1].keys():
                indeces = np.where(np.in1d(defaultValues[0][name], defaultValues[1][name]))[0]
                for index in indeces:
                    lstbox.select_set(index) 
        else:
            lstbox.select_set(0) 
        return lstbox
    
    def update_scrollregion(self, event, root):
        root.configure(scrollregion = root.bbox("all"))
        
    def run1(self, defaultVariable):
        self.output = dict()
        for k in defaultVariable.keys():
            self.output[k] = np.array(eval(defaultVariable[k].get()))
        self.root.destroy()
        
    def run2(self, timeVariable, Listboxes):
        # Select checkbuttons.
        timeSelect = [v for v in timeVariable.keys() if timeVariable[v].get()]
        # Select listbox.
        lagsSelect = dict()
        for k in Listboxes.keys():
            lagsSelect[k] = list()
            selection = Listboxes[k].curselection()
            for i in selection:
                lagsSelect[k].append(eval(Listboxes[k].get(i)))
        lagsSelect = {k: np.array(v) for k,v in lagsSelect.items() if v != []}
        if lagsSelect == {}:
            raise ValueError("Lag information must be provided for at least one indicator.")
        self.output = (timeSelect, lagsSelect)
        self.root.destroy()
        
    def run3(self, Variable):
        self.output = dict()
        for k in Variable.keys():
            self.output[k] = Variable[k].get()
        self.root.destroy()

    def GUI_lags_1(self, defaultLags):
        # Set label title into window.
        title = self.Label(self.frame, text = "Lags selection for indicators:", rowno = 0, colno = 0, size = 14)
        self.canvas.grid(row = 1, column = 0)
        # Set entries.
        defaultVariable = {k: tk.StringVar(value = str(list(v))) for k, v in defaultLags.items()}
        for i, (k, v) in enumerate(defaultLags.items()):
            # Set label.
            self.Label(self.canvas, text = k, rowno = 2+i, colno = 0, size = 9)
            # Set entry.
            self.EntryText(self.canvas, textVariable = defaultVariable[k], rowno = 2+i, colno = 1)
        # Set run button.
        runbutton = tk.Button(self.root, text = "Run", command = lambda: self.run1(defaultVariable))
        runbutton.grid(row = 2, column = 0)
        # Set label information.
        information = self.Label(self.root, text = "You can also provide python expressions (e.g. np.arange(1,15))", 
                                 rowno = 3, colno = 0, size = 10)
        
        self.root.mainloop()  
        
        return self.output
        
    def GUI_lags_2(self, allowedTimes, allowedLags, defaultTimes = None, defaultLags = None):
        self.canvas.grid(row = 0, column = 0)
        self.canvasFrame = tk.Frame(self.canvas)
        self.canvas.create_window(0, 0, window = self.canvasFrame, anchor = "nw")
        
        # Set label.
        lb = self.Label(self.canvasFrame, text = "Time features selection:", rowno = 0, colno = 0, size = 13, wraplength = 150)
        # Set checkbuttons (time variables).
        timeVariable = {v: tk.BooleanVar(value = True) if v in defaultTimes else tk.BooleanVar(value = False) for v in allowedTimes}
        for i,v in enumerate(allowedTimes):
            self.Checkbutton(self.canvasFrame, v, timeVariable[v], rowno = 0, colno = i+1)
        # Set label.
        lb = self.Label(self.canvasFrame, text = "Lags selection:", rowno = 1, colno = 0, size = 13, wraplength = 150)
        # Set listbox.
        lagsVariable = {k: tk.StringVar(value = tuple(v)) for k,v in allowedLags.items()}
        Listboxes = dict()
        for i, (k,v) in enumerate(allowedLags.items()):
            Listboxes[k] = self.Listbox(self.canvasFrame, k, lagsVariable[k], rowno = 1, colno = i+1, 
                                        defaultValues = (allowedLags, defaultLags))
        
        # Set button.
        btn = tk.Button(self.canvasFrame, text = "Run", command = lambda: self.run2(timeVariable, Listboxes))
        btn.grid(row = 2, column = 0)
        # Set label.
        lb = self.Label(self.frame, text = "N.B. If no time lag is selected, the corresponding indicator will not be taken into account as a predictor in the following analysis.", rowno = 3, colno = 0, size = 10)

        # Scroll horizontal.
        xscroll = tk.Scrollbar(self.frame, orient = tk.HORIZONTAL)
        xscroll.config(command = self.canvas.xview)
        self.canvas.config(xscrollcommand = xscroll.set)
        xscroll.grid(row = 2, column = 0, sticky = "ew")
        self.canvasFrame.bind("<Configure>", lambda x: self.update_scrollregion(x, self.canvas))

        self.root.mainloop()
        
        return self.output
    
    def GUI_indicators_1(self, indicators, target):
        # Set label.
        lb = self.Label(self.frame, text = "Select the indicators you want to consider:", 
                        rowno = 0, colno = 0, size = 12, wraplength = 300)
        self.canvas.config(width = 200, height = 250)
        self.canvas.grid(row = 1, column = 0)
        self.canvasFrame = tk.Frame(self.canvas)
        self.canvas.create_window(0, 0, window = self.canvasFrame, anchor = "nw")
        # Set checkbuttons.
        Variable = {v: tk.BooleanVar(value = True) for v in indicators}
        for i,v in enumerate(indicators):
            if v == target:
                self.Checkbutton(self.canvasFrame, v, Variable[v], rowno = i+2, colno = 0, disable = True)
            else:
                self.Checkbutton(self.canvasFrame, v, Variable[v], rowno = i+2, colno = 0)
        # Set button.
        btn = tk.Button(self.frame, text = "Run", command = lambda: self.run3(Variable))
        btn.grid(row = 3, column = 0)

        # Scroll vertical.
        yscroll = tk.Scrollbar(self.frame, orient = tk.VERTICAL)
        yscroll.config(command = self.canvas.yview)
        self.canvas.config(yscrollcommand = yscroll.set)
        yscroll.grid(row = 1, column = 1, sticky = "ns")
        self.canvasFrame.bind("<Configure>", lambda x: self.update_scrollregion(x, self.canvas))

        self.root.mainloop()
        
        return self.output
    
    def GUI_indicators_2(self, features, target):
        # Set label.
        lb = self.Label(self.frame, text = "Select the features on which to apply a feature selection:", 
                        rowno = 0, colno = 0, size = 12, wraplength = 300)
        self.canvas.config(width = 200, height = 250)
        self.canvas.grid(row = 1, column = 0)
        self.canvasFrame = tk.Frame(self.canvas)
        self.canvas.create_window(0, 0, window = self.canvasFrame, anchor = "nw")
        # Set checkbuttons.
        Variable = {v: tk.BooleanVar(value = False) for v in features}
        for i,v in enumerate(features):
            if v == target:
                self.Checkbutton(self.canvasFrame, v, Variable[v], rowno = i+2, colno = 0, disable = True)
            else:
                self.Checkbutton(self.canvasFrame, v, Variable[v], rowno = i+2, colno = 0)
        # Set button.
        btn = tk.Button(self.frame, text = "Run", command = lambda: self.run3(Variable))
        btn.grid(row = 3, column = 0)
        # Set label.
        lb = self.Label(self.frame, text = "N.B. If the feature is selected, it is applied a feature selection on that variable.", 
                        rowno = 4, colno = 0, size = 10, wraplength = 300)

        # Scroll vertical.
        yscroll = tk.Scrollbar(self.frame, orient = tk.VERTICAL)
        yscroll.config(command = self.canvas.yview)
        self.canvas.config(yscrollcommand = yscroll.set)
        yscroll.grid(row = 1, column = 1, sticky = "ns")
        self.canvasFrame.bind("<Configure>", lambda x: self.update_scrollregion(x, self.canvas))

        self.root.mainloop()
        
        return self.output
    
    