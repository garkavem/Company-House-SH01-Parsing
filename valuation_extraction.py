# -*- coding: utf-8 -*-
import pandas as pd
from pdf2image import convert_from_path
import cv2
import requests
import subprocess
from pathlib import Path
import re
import json
from datetime import datetime
import time
import configparser
import warnings

config = configparser.ConfigParser()
config.read('config.txt')
COMPANY_HOUSE_KEY = config['general']['CompanyHouseKey']
WORK_DIRECTORY = config['general']['Dir']


def send_request(ref):
    # waiting because of Companies House API Rate limit (600 requests within a five-minute period)
    try:
        res = requests.get(ref, headers={'Authorization': COMPANY_HOUSE_KEY}).json()
        return res
    except json.decoder.JSONDecodeError:
        print('...Zzz....', end=' ')
        time.sleep(20)
        return send_request(ref)


class DocumentProcessor:
    def __init__(self, doc_path):
        self.doc_path = doc_path
        self.form_type = 'unknown'
        self.date = pd.to_datetime('2020-01-01')

    def determine_form_type(self):
        def determine_from_text(text):
            if 'electronically filed document' in text.lower():
                if filing_date < datetime(2014, 1, 1):
                    self.form_type = 'online_old'
                else:
                    self.form_type = 'online'
                return
            if 'version6.0' in text.lower().replace(' ', ''):
                self.form_type = 'offline6'
                return
            if 'version5.0' in text.lower().replace(' ', ''):
                self.form_type = 'offline5'
                return
            if 'version4.0' in text.lower().replace(' ', ''):
                self.form_type = 'offline4'
                return

        with open(self.doc_path + '/metadata.json', 'r') as f:
            filing_date = pd.to_datetime(json.load(f)['date'])

        if Path(self.doc_path + 'pages/form_type.txt').is_file():
            with open(self.doc_path + 'pages/form_type.txt', 'r') as f:
                detected_text = f.read()
            return determine_from_text(detected_text)

        for page in range(3):
            img = cv2.imread(self.doc_path + 'pages/{}.jpeg'.format(page))
            if img is None:
                continue
            crop_img = img[8 * img.shape[0] // 10:, :]
            cv2.imwrite(self.doc_path + 'pages/formtype.jpg', crop_img)
            tesseract_command = 'tesseract {} {} --psm 6'.format(self.doc_path + 'pages/formtype.jpg',
                                                                 self.doc_path + 'pages/form_type')
            subprocess.call(tesseract_command, shell=True)
            with open(self.doc_path + 'pages/form_type.txt', 'r') as f:
                detected_text = f.read()
            determine_from_text(detected_text)
            if self.form_type != 'unknown':
                return

    @staticmethod
    def remove_table_borders(image):
        result = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Remove horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        remove_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        contours = cv2.findContours(remove_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        for c in contours:
            cv2.drawContours(result, [c], -1, (255, 255, 255), 5)

        # Remove vertical lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        remove_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        contours = cv2.findContours(remove_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        for c in contours:
            cv2.drawContours(result, [c], -1, (255, 255, 255), 5)
        return result

    def get_text_from_image(self, img_name, psm):
        if not Path(self.doc_path + 'pages/{}.txt'.format(img_name)).is_file():
            tesseract_command = 'tesseract {} {} --psm {}'.format(self.doc_path + 'pages/{}.jpg'.format(img_name),
                                                                  self.doc_path + 'pages/' + img_name, psm)

            subprocess.call(tesseract_command, shell=True)

        with open(self.doc_path + 'pages/{}.txt'.format(img_name), 'r') as f:
            detected_text = f.read().replace(',', '').replace('-', '').replace('—', '').replace(
                '_', '').replace('!', '').replace('|', '').replace(')', '').lower()
            detected_text = re.sub(r' +', ' ', detected_text)
            detected_text = re.sub(r'(\d) \. (\d)', r'\1.\2', detected_text)
            detected_text = re.sub('pound sterling', '£', detected_text)
            detected_text = re.sub('usd', '\$', detected_text)
            detected_text = re.sub(r'(€|\$|£|eur|gbp) ', r'\1', detected_text)
            detected_text = re.sub(r'\n+', r'\n', detected_text)
        return detected_text

    def process_currencies_share_price(self, price_share):
        if price_share == 'nil':
            return None
        if '$' in price_share:
            price_share = float(price_share.replace('$', '').replace('us', ''))
            rate = requests.get(
                'https://api.exchangeratesapi.io/{:%Y-%m-%d}?base=USD'.format(self.date)).json()
            price_share *= rate['rates']['GBP']

        elif '€' in price_share or 'eur' in price_share:
            price_share = float(price_share.replace('€', '').replace('eur', ''))
            rate = requests.get(
                'https://api.exchangeratesapi.io/{:%Y-%m-%d}?base=EUR'.format(self.date)).json()
            price_share *= rate['rates']['GBP']

        else:
            price_share = float(price_share.replace('£', '').replace('gbp', ''))

        return price_share

    def extract_share_price_n_allotted_online(self):
        if not Path(self.doc_path + 'pages/share_price.txt').is_file():
            img = cv2.imread(self.doc_path + 'pages/0.jpeg')
            crop_img = img[39 * img.shape[0] // 100: 9 * img.shape[0] // 10, :]
            cv2.imwrite(self.doc_path + 'pages/0cropped.jpg', crop_img)

            tesseract_command = 'tesseract {} {} --psm 6'.format(self.doc_path + 'pages/0cropped.jpg',
                                                                 self.doc_path + 'pages/share_price')
            subprocess.call(tesseract_command, shell=True)

        with open(self.doc_path + 'pages/share_price.txt', 'r') as f:
            detected_text = f.read()

        if 'amount paid' not in detected_text.lower():
            return None, None

        replace_dict = {':': '',
                        ' ': '',
                        '/': '7',
                        '§': '5',
                        '|': '',
                        "'": '',
                        '©': '0',
                        '—': ''}
        price_share = detected_text.lower().split('amount paid')[1].split('\n')[0].strip()
        n_allotted = detected_text.lower().split('number allotted')[1].split('\n')[0].strip()

        for replaced_symbol in replace_dict.keys():
            if replaced_symbol in price_share:
                price_share = price_share.replace(replaced_symbol, replace_dict[replaced_symbol]).strip()
            if replaced_symbol in n_allotted:
                n_allotted = n_allotted.replace(replaced_symbol, replace_dict[replaced_symbol]).strip()

        if '$' in n_allotted:
            n_allotted = n_allotted.replace('$', '8')

        return float(price_share), float(n_allotted)

    def extract_share_price_n_allotted_offline6(self):
        if not Path(self.doc_path + 'pages/{}.jpg'.format('0cropped')).is_file():
            img = cv2.imread(self.doc_path + 'pages/0.jpeg')
            crop_img = img[5 * img.shape[0] // 10: 9 * img.shape[0] // 10, :]

            no_borders_img = self.remove_table_borders(crop_img)

            cv2.imwrite(self.doc_path + 'pages/{}.jpg'.format('0cropped'), no_borders_img)
        detected_text = self.get_text_from_image('0cropped', 6)
        detected_text = detected_text.split('currency')[1]

        reg = re.search(r"(\d(\n)?\s?\.?£?\$?€?(\n)?){6}", detected_text)

        if reg is None:
            detected_text = self.get_text_from_image('0cropped', 11)
            reg = re.search(r"(\d(\n)?\s?\.?£?\$?€?(\n)?){6}", detected_text)
            if reg is None:
                return None, None

        table_line = detected_text[reg.span()[0]:].replace('|', ' ').replace('\\', '')

        price_share = table_line.split()[2].replace('©', '0')
        n_allotted = table_line.split()[0].replace('©', '0')

        n_allotted = float(n_allotted)

        price_share = self.process_currencies_share_price(price_share)

        return price_share, n_allotted

    def extract_share_price_n_allotted(self):
        if self.form_type == 'online' or self.form_type == 'online_old':
            return self.extract_share_price_n_allotted_online()

        if self.form_type == 'offline6':
            return self.extract_share_price_n_allotted_offline6()

        if self.form_type == 'offline5' or self.form_type == 'offline4':
            return self.extract_share_price_n_allotted_offline6()
        return None, None

    def extract_total_shares_online_form(self):
        if Path(self.doc_path + 'pages/n_shares.txt').is_file():
            with open(self.doc_path + 'pages/n_shares.txt', 'r') as f:
                detected_text = f.read()
            if 'total number of shares' not in detected_text.lower():
                return None
            total_shares = (
                detected_text.lower().split('total number of shares')[1].split('\n')[0].replace(':', '').replace(
                    '/', '7').replace('§', '5').replace(' ', '').replace("'", ''))
            return total_shares

        for page in range(2, 10):
            page_file = self.doc_path + 'pages/{}.jpeg'.format(page)
            img = cv2.imread(page_file)
            if img is None:
                continue
            crop_img = img[: 5 * img.shape[0] // 10, :]

            cv2.imwrite(self.doc_path + 'pages/2cropped.jpg', crop_img)
            tesseract_command = 'tesseract {} {} --psm 6'.format(self.doc_path + 'pages/2cropped.jpg',
                                                                 self.doc_path + 'pages/n_shares')

            subprocess.call(tesseract_command, shell=True)

            with open(self.doc_path + 'pages/n_shares.txt', 'r') as f:
                detected_text = f.read()
            if 'total number of shares' not in detected_text.lower():
                continue
            total_shares = float(
                detected_text.lower().split('total number of shares')[1].split('\n')[0].replace(':', '').replace(
                    '/', '7').replace('§', '5').replace(' ', '').replace("'", ''))
            return total_shares

    def extract_total_shares_online_old_form(self):
        for page in range(1, 10):
            if not Path(self.doc_path + 'pages/n_shares.txt').is_file():
                page_file = self.doc_path + 'pages/{}.jpeg'.format(page)
                img = cv2.imread(page_file)
                if img is None:
                    continue
                crop_img = img[:, :]

                cv2.imwrite(self.doc_path + 'pages/2cropped.jpg', crop_img)
                tesseract_command = 'tesseract {} {} --psm 4'.format(self.doc_path + 'pages/2cropped.jpg',
                                                                     self.doc_path + 'pages/n_shares')

                subprocess.call(tesseract_command, shell=True)

            with open(self.doc_path + 'pages/n_shares.txt', 'r') as f:
                detected_text = f.read()
            if 'statement of capital (totals)' not in detected_text.lower():
                continue
            try:
                total_shares = (
                    detected_text.lower().split('total number')[1].split('of shares')[0].replace(':', '').replace(
                        '/', '7').replace('§', '5').replace(' ', '').replace("'", '').replace('\n', ' '))
            except IndexError:
                return None
            return total_shares

    def extract_total_shares_offline6(self):
        def extract_from_text(text):
            if 'ist total aggregate' not in text:
                return None
            reg = re.search(r"(\d(\n)?\s?\.?£?\$?€?(\n)?){6}", text)

            if reg is None:
                text = self.get_text_from_image('cropped2', 11)
                reg = re.search(r"(\n|\s)\d\d{2,15}(\n|\s)", text)
                if reg is None:
                    return None
                total_sh = text[reg.span()[0]:reg.span()[1]].replace(' ', '').replace('\n', ' ').split()[0]
            else:
                table_line = text[reg.span()[0]:].replace('|', ' ')
                total_sh = table_line.split()[0]
            return total_sh

        if Path(self.doc_path + 'pages/{}.jpg'.format('2cropped')).is_file():
            detected_text = self.get_text_from_image('2cropped', 4)
            return extract_from_text(detected_text)

        for page in range(1, 4):
            page_file = self.doc_path + 'pages/{}.jpeg'.format(page)
            img = cv2.imread(page_file)
            crop_img = img[5 * img.shape[0] // 10: 9 * img.shape[0] // 10, :]

            no_borders_img = self.remove_table_borders(crop_img)

            cv2.imwrite(self.doc_path + 'pages/{}.jpg'.format('2cropped'), no_borders_img)
            detected_text = self.get_text_from_image('2cropped', 4)
            total_shares = extract_from_text(detected_text)
            if total_shares is not None:
                return total_shares
        return None

    def extract_total_shares_offline5(self):
        if not Path(self.doc_path + 'pages/{}.jpg'.format('2cropped')).is_file():
            page_file = self.doc_path + 'pages/{}.jpeg'.format(1)
            img = cv2.imread(page_file)
            crop_img = img[: 5 * img.shape[0] // 10, :]

            cv2.imwrite(self.doc_path + 'pages/{}.jpg'.format('2cropped'), crop_img)
        detected_text = self.get_text_from_image('2cropped', 4)
        reg = re.search('totals\s?\|?\s?\d\d', detected_text)
        if reg is None:
            return None
        table_line = detected_text[reg.span()[0] + 6:].replace('|', ' ').replace(',', '')
        return table_line.split()[0]

    def extract_total_shares(self):
        total_shares = None
        if self.form_type == 'online_old':
            total_shares = self.extract_total_shares_online_old_form()

        if self.form_type == 'online':
            total_shares = self.extract_total_shares_online_form()

        if self.form_type == 'offline6':
            total_shares = self.extract_total_shares_offline6()

        if self.form_type == 'offline5' or self.form_type == 'offline4':
            total_shares = self.extract_total_shares_offline5()

        try:
            total_shares = float(total_shares)
            return total_shares
        except ValueError:
            return None
        except TypeError:
            return None

    def run(self):
        date = self.doc_path.split('/')[-2].split('_')[0]
        self.date = date
        self.determine_form_type()

        try:
            share_price, n_allotted = self.extract_share_price_n_allotted()
        except Exception:
            share_price, n_allotted = None, None

        total_shares = self.extract_total_shares()

        if n_allotted is not None and total_shares is not None and share_price is not None:
            fundraising = n_allotted * share_price
            valuation = total_shares * share_price
            if valuation == 0:
                equity = None
            else:
                equity = fundraising / valuation
        else:
            fundraising = None
            valuation = None
            equity = None

        with open(self.doc_path + '/metadata.json', 'r') as f:
            metadata = json.load(f)
            if 'capital' in metadata['description_values']:
                capital = metadata['description_values']['capital'][0]
                capital['figure'] = float(capital['figure'].replace(',', ''))
            else:
                capital = None
            transaction_id = metadata['transaction_id']

        d = {'date': date, 'form_type': self.form_type, 'share_price': share_price, 'n_allotted': n_allotted,
             'total_shares': total_shares,
             'fundraising': fundraising, 'valuation': valuation, 'equity': equity,
             'capital': capital, 'transaction_id': transaction_id}
        return d


def parse_document(doc_item, ch_id):
    doc_folder_path = WORK_DIRECTORY + '/' + ch_id
    doc_path = '{}/{}_{}/'.format(doc_folder_path, doc_item['action_date'], doc_item['transaction_id'])
    doc_proc = DocumentProcessor(doc_path)
    return doc_proc.run()


def download_document(doc_item, ch_id):
    doc_folder_path = WORK_DIRECTORY + '/' + ch_id
    if 'links' not in doc_item:
        return
    if 'document_metadata' not in doc_item['links']:
        return
    doc_path = '{}/{}_{}/'.format(doc_folder_path, doc_item['action_date'], doc_item['transaction_id'])
    if Path(doc_path + 'document.pdf').is_file():
        warnings.warn('SH01 document {:} already downloaded. Download skipped.'.format(doc_item['transaction_id']))
        return

    document_id = doc_item['links']['document_metadata'].split('/')[-1]
    curl_command = 'curl -i -u{}: https://document-api.companieshouse.gov.uk/document/{}/content'
    doc_content_res = subprocess.check_output(curl_command.format(COMPANY_HOUSE_KEY, document_id), shell=True)
    aws_url = str(doc_content_res).split('Location: ')[-1].split('\\r\\nServer:')[0]
    aws_response = requests.get(aws_url)
    Path(doc_path + 'pages/').mkdir(parents=True, exist_ok=True)

    with open(doc_path + 'metadata.json', 'w') as f:
        json.dump(doc_item, f)

    with open(doc_path + 'document.pdf', 'wb') as f:
        f.write(aws_response.content)

    pages = convert_from_path(doc_path + '/document.pdf', 500, last_page=10)
    for idx, page in enumerate(pages[:10]):
        page.save('{}/pages/{}.jpeg'.format(doc_path, idx), 'JPEG')


def get_filing_history(ch_id):
    fh_req = 'https://api.companieshouse.gov.uk/company/{}/filing-history?start_index={}&items_per_page=100'
    n_items = 100
    start_index = 0
    sh01_docs = []
    while n_items == 100:
        response = send_request(fh_req.format(ch_id, start_index))
        n_items = len(response.get('items', []))
        start_index += n_items
        sh01_docs.extend([i for i in response.get('items', []) if i.get('type', '') == 'SH01'])
    return sh01_docs


def process_ch_id(ch_id):
    sh01_docs = get_filing_history(ch_id)
    for doc in sh01_docs:
        download_document(doc, ch_id)
        res = parse_document(doc, ch_id)
        doc_folder_path = WORK_DIRECTORY + '/' + ch_id
        doc_path = '{}/{}_{}/'.format(doc_folder_path, doc['action_date'], doc['transaction_id'])
        with open(doc_path + 'result.json', 'w') as f:
            json.dump(res, f)
    return


def process_ch_ids_list(ch_list_path=WORK_DIRECTORY + '/company_house_ids_list'):
    with open(ch_list_path, 'r') as f:
        ch_ids = f.readlines()
    ch_ids = [i.replace('\n', '') for i in ch_ids]
    for ch_id in ch_ids:
        process_ch_id(ch_id)
    return
