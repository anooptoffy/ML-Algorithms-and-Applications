#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
ZetCode Tkinter tutorial

In this script, we use the grid
manager to create a more complicated
layout.

Author: Jan Bodnar
Last modified: December 2015
Website: www.zetcode.com

code  : http://zetcode.com/gui/tkinter/layout/
        https://www.tutorialspoint.com/python/tk_panedwindow.htm
"""

from tkinter import Tk, Text, BOTH, W, N, E, S
from ttk import Frame, Button, Label, Style


class UINaive(Frame):
    def __init__(self, parent):
        Frame.__init__(self, parent)

        self.parent = parent
        self.initUI()

    def initUI(self):
        self.parent.title("Windows")
        self.pack(fill=BOTH, expand=True)

        self.columnconfigure(1, weight=1)
        self.columnconfigure(3, pad=7)
        self.rowconfigure(3, weight=1)
        self.rowconfigure(5, pad=7)

        lbl = Label(self, text="Windows")
        lbl.grid(sticky=W, pady=4, padx=5)

        area = Text(self)
        area.grid(row=1, column=0, columnspan=2, rowspan=4,
                  padx=5, sticky=E + W + S + N)

        abtn = Button(self, text="Select Folder")
        abtn.grid(row=1, column=4)

        cbtn = Button(self, text="Close")
        cbtn.grid(row=2, column=4, pady=4)

        hbtn = Button(self, text="Help")
        hbtn.grid(row=5, column=0, padx=5)

        obtn = Button(self, text="OK")
        obtn.grid(row=5, column=4)


def main():
    root = Tk()
    root.geometry("900x900")
    app = UINaive(root)
    root.mainloop()


if __name__ == '__main__':
    main()