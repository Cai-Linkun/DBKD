import requests
import time
import pickle
import logging
import sys
import os
import csv

from bs4 import BeautifulSoup

MAIN_URL = "http://www.icd9data.com"
HIERARCHY = []
SAVE_FILE = "hierarchy.bin"
SAVE_DIR = "./processed_data"

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler(stream=sys.stdout)
file_handler = logging.FileHandler("download.log", delay=0, encoding="utf8")
console_handler.setLevel(logging.DEBUG)
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)

if os.path.exists(SAVE_FILE):
    with open(SAVE_FILE, "rb") as f:
        HIERARCHY = pickle.load(f)
        logger.info("load {} records".format(len(HIERARCHY)))


def Append(hierarchy):
    HIERARCHY.append(hierarchy)


def WriteToFile():
    with open("hierarchy.bin", "wb") as f:
        pickle.dump(HIERARCHY, f)


# Function to fetch the page content
def fetch_page(url):
    text = ""

    def _fetch(url):
        response = requests.get(url)
        response.raise_for_status()  # Ensure we got a valid response
        return response.text

    RETRY_TIMES = 5
    for i in range(RETRY_TIMES):
        try:
            text = _fetch(url)
        except Exception as e:
            logger.warning("retrying: {}".format(url))
            time.sleep(5 * (i + 1))
            continue
        else:
            break
    # time.sleep(2)
    return text


# Function to extract first-level codes
def get_first_level_codes(soup):
    first_level_codes = []
    first_level_div = soup.find("div", class_="definitionList")
    if first_level_div:
        first_level_codes = [
            (a.text.strip(), MAIN_URL + a["href"])
            for a in first_level_div.find_all("a")
        ]
    return first_level_codes


# Function to extract second-level codes from a first-level page
def get_second_level_codes(url):
    second_level_codes = []
    page = fetch_page(url)
    soup = BeautifulSoup(page, "html.parser")
    second_level_ul = soup.find("div", class_="definitionList")
    if second_level_ul:
        second_level_codes = [
            (a.text.strip(), MAIN_URL + a["href"])
            for a in second_level_ul.find_all("a", {"class": "identifier"})
        ]
    else:
        second_level_ul = soup.find("ul", class_="definitionList")
        if second_level_ul:
            second_level_codes = [
                (a.text.strip(), MAIN_URL + a["href"])
                for a in second_level_ul.find_all("a", {"class": "identifier"})
            ]
    return second_level_codes


# Function to extract third-level codes from a second-level page
def get_third_level_codes(url):
    third_level_codes = []
    page = fetch_page(url)
    soup = BeautifulSoup(page, "html.parser")
    third_level_ul = soup.find("ul", class_="definitionList")
    if third_level_ul:
        third_level_codes = [
            (a.text.strip(), MAIN_URL + a["href"])
            for a in third_level_ul.find_all("a", {"class": "identifier"})
        ]
    else:
        third_level_ul = soup.find("ul", class_="codeHierarchyUL")
        if third_level_ul:
            third_level_codes = [
                (a.text.strip(), MAIN_URL + a["href"])
                for a in third_level_ul.find_all("a", {"class": "identifier"})
            ]
    return third_level_codes


def get_four_level_codes(url):
    four_level_codes = []
    page = fetch_page(url)
    soup = BeautifulSoup(page, "html.parser")
    four_level_ul = soup.find("ul", class_="codeHierarchyUL")
    if four_level_ul:
        four_level_codes = [
            a.text.strip() for a in four_level_ul.find_all("a", class_="identifier")
        ]
    return four_level_codes


# Function to extract the full hierarchy [first, second, third] for each level
def scrape_hierarchy(base_url):
    full_hierarchy = []

    # Step 1: Get first-level codes
    main_page = fetch_page(base_url)
    soup = BeautifulSoup(main_page, "html.parser")
    first_level_codes = get_first_level_codes(soup)

    for first_code, first_url in first_level_codes:
        # 280-289 740-759
        if first_code not in [
            "280-289",
            "740-759",
        ]:
            logger.info("{} already processed".format(first_code))
            continue
        logger.info("processing first code: {}".format(first_code))
        # Step 2: Get second-level codes for each first-level code
        second_level_codes = get_second_level_codes(first_url)
        if len(second_level_codes) == 0:
            logger.warning(
                "no second level, code: {}, url: {}".format(first_code, first_url)
            )
            Append([first_code])

        for second_code, second_url in second_level_codes:
            logger.info(" processing second code: {}".format(second_code))
            # Step 3: Get third-level codes for each second-level code
            third_level_codes = get_third_level_codes(second_url)

            if len(third_level_codes) == 0:
                logger.warning(
                    "no third level, code: {}, url: {} ".format(second_code, second_url)
                )
                Append([first_code, second_code])

            for third_code, third_url in third_level_codes:
                logger.info("   processing third code: {}".format(third_code))
                # Append the full hierarchy: [first level, second level, third level]
                four_level_codes = get_four_level_codes(third_url)
                if len(four_level_codes) == 0:
                    logger.warning(
                        "no four level, code: {}, url: {} ".format(
                            third_code, third_url
                        )
                    )
                    Append([first_code, second_code, third_code])

                for four_code in four_level_codes:
                    Append([first_code, second_code, third_code, four_code])

    return full_hierarchy


def download():
    try:
        base_url = MAIN_URL + "/2012/Volume1/default.htm"
        hierarchy = scrape_hierarchy(base_url)
    except Exception as err:
        logger.error(str(err))
        WriteToFile()
    else:
        WriteToFile()


def preprocess():
    writed = []
    HIERARCHY
    # Writing to a CSV file
    with open(os.path.join(SAVE_DIR, "label_hierarchy.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        # Write each list (row) into the CSV file
        for line in HIERARCHY:
            if line not in writed:
                writed.append(line)
                writer.writerow(line)
            else:
                logger.warning("duplicated {}".format(line))


if __name__ == "__main__":
    # download()
    preprocess()
