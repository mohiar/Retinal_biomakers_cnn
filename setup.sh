#!/bin/bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
echo "✅ Virtual env setup complete!"
echo "📁 Next: unzip Kaggle dataset to 'oct2017/' folder (train/test/val structure)"
