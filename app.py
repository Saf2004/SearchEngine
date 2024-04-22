from flask import Flask, render_template

import os
os.environ["JAVA_HOME"] = "/Library/Java/JavaVirtualMachines/jdk-22.jdk/Contents/Home"


import pyterrier as pt
if not pt.started():
    pt.init()

app = Flask(__name__)
@app.route('/')
def hello_world():
    return render_template('homePage.html')


if __name__ == '__main__':
    app.run()

