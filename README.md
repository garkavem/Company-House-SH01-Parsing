# Company-House-SH01-Parsing

This code does the following for the companies which Companies House ids are put in data/companies_house_ids_list.txt:

* Gets the filing history via Companies House API.
* Downloads all the documents of SH01 type.
* Reads them using Tesseract Open Source OCR Engine to get the number of shares allotted, share price and total amount of shares. These values in turn allow to infer the funding amount and the company valuation.
