# Company-House-SH01-Parsing

This code does the following for the companies which Companies House ids are put in data/companies_house_ids_list.txt:

* Gets the filing history via Companies House API.
* Downloads all the documents of SH01 type.
* Reads them using Tesseract Open Source OCR Engine to get the number of shares allotted, share price and total amount of shares. These values in turn allow to infer the funding amount and the company valuation.

### Accuracy

There are different versions of document form in the database. The accuracy of our parcing method differs significantly for the documents of different types. So I estimated accuracy for each document type separately. 

|            |   form_type | errors/checked documents |
|:-----------|------------:|------------|
| online     |        3832 |0/50
| offline6   |         868 |3/20
| online_old |         258 |6/20
| offline5   |         188 |4/20
| offline4   |           5 |
