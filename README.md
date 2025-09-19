# iOS-Duplicate-Image-Finder

iOS Duplicate Image Finder

A tool to detect and manage duplicate or similar images in Xcode .xcassets folders.

In large iOS projects, especially in enterprise environments, it is common for duplicate images to accumulate across asset catalogs. This can silently bloat app size, clutter repositories, and waste developer time deciding which asset should be kept.

This project provides a simple solution: upload a zipped .xcassets folder and instantly identify duplicate or near-duplicate images with similarity scoring and report generation.

TRY IT ONLINE

You don’t need to run this tool via terminal. Use the hosted version here:
https://ios-similar-images.streamlit.app/

FEATURES

Detects and groups duplicate/similar images in .xcassets

Slider to set minimum similarity threshold

Sorting by similarity or image size

Paginated results (10 images per page) with suggested “keeper” image

Export results as an HTML report

Parallelized duplicate checking for faster performance on large codebases

DEMO

https://github.com/user-attachments/assets/ada508b6-e291-45a7-9d31-96bafa0b6b5e

LOCAL SETUP (OPTIONAL)

If you prefer running locally instead of using the hosted version:

Clone the repository
git clone https://github.com/kevinabram111/iOS-Duplicate-Image-Finder.git

Change into the project directory
cd iOS-Duplicate-Image-Finder

Install dependencies
pip install -r requirements.txt

Run with Streamlit
streamlit run ios_similar_images.py

Then open the provided localhost link in your browser.

BACKGROUND

This project was inspired by challenges faced in large-scale iOS development at CIMB Niaga, and connects with concepts from the Big Data course at Monash University. It highlights how small, practical tools can bring real impact in maintaining clean, efficient codebases.

LICENSE

MIT License — feel free to use, modify, and share.
