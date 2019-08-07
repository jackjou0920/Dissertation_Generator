#!/usr/bin/env python

"""
Converts PDF text content (though not images containing text) to plain text, html, xml or "tags".
"""
import argparse
import logging
import six
import sys
import re
import pdfminer.settings
import os
pdfminer.settings.STRICT = False
import pdfminer.high_level
import pdfminer.layout
import MySQLdb
from pdfminer.image import ImageWriter
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.pdfpage import PDFPage
from io import StringIO


def extract_text(files=[], outfile='-',
            _py2_no_more_posargs=None,  # Bloody Python2 needs a shim
            no_laparams=False, all_texts=None, detect_vertical=None, # LAParams
            word_margin=None, char_margin=None, line_margin=None, boxes_flow=None, # LAParams
            output_type='text', codec='utf-8', strip_control=False,
            maxpages=0, page_numbers=None, password="", scale=1.0, rotation=0,
            layoutmode='normal', output_dir=None, debug=False,
            disable_caching=False, **other):
    if _py2_no_more_posargs is not None:
        raise ValueError("Too many positional arguments passed.")
    if not files:
        raise ValueError("Must provide files to work upon!")

    # If any LAParams group arguments were passed, create an LAParams object and
    # populate with given args. Otherwise, set it to None.
    if not no_laparams:
        laparams = pdfminer.layout.LAParams()
        for param in ("all_texts", "detect_vertical", "word_margin", "char_margin", "line_margin", "boxes_flow"):
            paramv = locals().get(param, None)
            if paramv is not None:
                setattr(laparams, param, paramv)
    else:
        laparams = None

    imagewriter = None
    if output_dir:
        imagewriter = ImageWriter(output_dir)

    if output_type == "text" and outfile != "-":
        for override, alttype in ((".htm", "html"), (".html", "html"), (".xml", "xml"), (".tag", "tag") ):
            if outfile.endswith(override):
                output_type = alttype

    if outfile == "-":
        outfp = sys.stdout
        if outfp.encoding is not None:
            codec = 'utf-8'
    else:
        outfp = open(outfile, "wb")

    for fname in files:
        with open(fname, "rb") as fp:
            pdfminer.high_level.extract_text_to_fp(fp, **locals())

    return outfp

#======================================================================================================================
def extract_topic(head):
    text = [l for l in head.splitlines() if l.strip()]
    #print(text)

    topic = ""
    for i in range(len(text)):
        if ("supervisor" in text[i+1]) or ("supervisors" in text[i+1]) or ("supervised" in text[i+1]):
            break
        if ("dr." in text[i]) or ("author" in text[i]) or ("student" in text[i]):
            break
        if ("report is submitted" in text[i]):
            break
        regexp = re.compile(r'by(\s)*\:')
        if regexp.match(text[i]):
            break
        regexp = re.compile(r'com(\s)*\d\d\d\d')
        if regexp.match(text[i]):
            break
        if ("university" in text[i]) or ("sheffield" in text[i]):
            topic = ""
            continue
        if ("master" in text[i]) or ("dissertation" in text[i]):
            topic = ""
            continue
        if ("department" in text[i]) or ("computer science" in text[i]):
            topic = ""
            continue
        topic += text[i] + " "
    return topic

#======================================================================================================================
def preprocessing(fname, head, str):
    head = head.lower()
    str = str.lower()
    topic = extract_topic(head)
    print("topic:")
    print(topic)
    print()

    text = [l for l in str.splitlines() if l.strip()]
    #print(text)
    is_abstract = False
    is_constents = False
    is_intro = False
    abstract = ""
    introduction = ""
    for i in range(len(text)):
        if (("ii" == text[i]) or ("ii " == text[i]) or ("iii" == text[i]) or ("iii " == text[i]) or \
            ("i" == text[i]) or ("i " == text[i]) or ("v" == text[i]) or ("v " == text[i]) or \
            ("2" == text[i]) or ("2 " == text[i]) or ("3" == text[i]) or ("3 " == text[i])) and (is_abstract == True):
            is_abstract = False
            break
        if (("acknowledgements" in text[i]) or ("acknowledgement" in text[i]) or \
            ("contents" in text[i]) or ("content" in text[i]) or ("acronyms" in text[i])) and (is_abstract == True):
            is_abstract = False
            break
        # if (("references" in text[i]) or ("appendices" in text[i])) and (is_constents == True):
        #     is_constents = False
        # if (("chapter 2" in text[i]) or ("literature review" in text[i])) and (is_intro == True):
        #     is_intro = False
        #     break

        if (is_abstract):
            abstract += text[i] + '\n'
            if ("abstract" in text[i]) or ("ABSTRACT" in text[i]):
                abstract = ""
        # if (is_intro):
        #     introduction += text[i] + '\n'


        if ("abstract" in text[i]) or ("ABSTRACT" in text[i]):
            is_abstract = True
        # if ("contents" in text[i]):
        #     is_constents = True
        # if ("introduction" in text[i]) and (is_constents == False):
        #     is_intro = True

    # abstract = abstract.rstrip('\n')
    #
    # print("abstract:")
    # print(abstract)
    # print()
    # save_to_db(fname, topic, abstract)

    # print("introduction:")
    # print(introduction)

#======================================================================================================================
def save_to_db(fname, topic, abstract):
    db = MySQLdb.connect(host="143.167.8.208", user="root", passwd="jack", db="dissertation")
    cursor = db.cursor()

    sql = "INSERT INTO content(filename,topic,abstract) VALUES(%s,%s,%s)"
    try:
        cursor.execute(sql,(fname,topic,abstract))
        db.commit()
    except MySQLdb.Error as e:
        print(e)

#======================================================================================================================
def convert_pdf_to_txt(files=[], outfile='-',
            _py2_no_more_posargs=None,  # Bloody Python2 needs a shim
            no_laparams=False, all_texts=None, detect_vertical=None, # LAParams
            word_margin=None, char_margin=None, line_margin=None, boxes_flow=None, # LAParams
            output_type='text', codec='utf-8', strip_control=False,
            maxpages=0, page_numbers=None, password="", scale=1.0, rotation=0,
            layoutmode='normal', output_dir=None, debug=False,
            disable_caching=False, **other):

    # If any LAParams group arguments were passed, create an LAParams object and
    # populate with given args. Otherwise, set it to None.
    if not no_laparams:
        laparams = pdfminer.layout.LAParams()
        for param in ("all_texts", "detect_vertical", "word_margin", "char_margin", "line_margin", "boxes_flow"):
            paramv = locals().get(param, None)
            if paramv is not None:
                setattr(laparams, param, paramv)
    else:
        laparams = None

    # dirPath = "./"
    # files = [f for f in os.listdir(dirPath) if os.path.isfile(os.path.join(dirPath, f))]
    # files = sorted(files)
    # for fname in files:
        # if ("pdf" not in fname):
        #     continue
        # print(fname)
    with open('./previous/acp11vl.pdf', "rb") as fp:
        rsrcmgr = PDFResourceManager()
        retstr = StringIO()
        device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        pagenos=set()

            # for page in PDFPage.get_pages(fp, pagenos, maxpages=1, password=password, caching=True, check_extractable=True):
            #     interpreter.process_page(page)
            # head = retstr.getvalue()

        for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password, caching=True, check_extractable=True):
            interpreter.process_page(page)
        str = retstr.getvalue()

            # preprocessing(fname, head, str)
        print(str)
        device.close()
        retstr.close()


#======================================================================================================================
def maketheparser():
    parser = argparse.ArgumentParser(description=__doc__, add_help=True)
    #parser.add_argument("files", type=str, default=None, nargs="+", help="File to process.")
    parser.add_argument("-d", "--debug", default=False, action="store_true", help="Debug output.")
    parser.add_argument("-p", "--pagenos", type=str, help="Comma-separated list of page numbers to parse. Included for legacy applications, use --page-numbers for more idiomatic argument entry.")
    parser.add_argument("--page-numbers", type=int, default=None, nargs="+", help="Alternative to --pagenos with space-separated numbers; supercedes --pagenos where it is used.")
    parser.add_argument("-m", "--maxpages", type=int, default=0, help="Maximum pages to parse")
    parser.add_argument("-P", "--password", type=str, default="", help="Decryption password for PDF")
    parser.add_argument("-o", "--outfile", type=str, default="-", help="Output file (default \"-\" is stdout)")
    parser.add_argument("-t", "--output_type", type=str, default="text", help="Output type: text|html|xml|tag (default is text)")
    parser.add_argument("-c", "--codec", type=str, default="utf-8", help="Text encoding")
    parser.add_argument("-s", "--scale", type=float, default=1.0, help="Scale")
    parser.add_argument("-A", "--all-texts", default=None, action="store_true", help="LAParams all texts")
    parser.add_argument("-V", "--detect-vertical", default=None, action="store_true", help="LAParams detect vertical")
    parser.add_argument("-W", "--word-margin", type=float, default=None, help="LAParams word margin")
    parser.add_argument("-M", "--char-margin", type=float, default=None, help="LAParams char margin")
    parser.add_argument("-L", "--line-margin", type=float, default=None, help="LAParams line margin")
    parser.add_argument("-F", "--boxes-flow", type=float, default=None, help="LAParams boxes flow")
    parser.add_argument("-Y", "--layoutmode", default="normal", type=str, help="HTML Layout Mode")
    parser.add_argument("-n", "--no-laparams", default=False, action="store_true", help="Pass None as LAParams")
    parser.add_argument("-R", "--rotation", default=0, type=int, help="Rotation")
    parser.add_argument("-O", "--output-dir", default=None, help="Output directory for images")
    parser.add_argument("-C", "--disable-caching", default=False, action="store_true", help="Disable caching")
    parser.add_argument("-S", "--strip-control", default=False, action="store_true", help="Strip control in XML mode")
    return parser


#======================================================================================================================
def main(args=None):
    # P = maketheparser()
    # A = P.parse_args(args=args)
    #
    # if A.page_numbers:
    #     A.page_numbers = set([x-1 for x in A.page_numbers])
    # if A.pagenos:
    #     A.page_numbers = set([int(x)-1 for x in A.pagenos.split(",")])
    #
    # imagewriter = None
    # if A.output_dir:
    #     imagewriter = ImageWriter(A.output_dir)
    #
    # if six.PY2 and sys.stdin.encoding:
    #     A.password = A.password.decode(sys.stdin.encoding)
    #
    # if A.output_type == "text" and A.outfile != "-":
    #     for override, alttype in ((".htm",  "html"), (".html", "html"), (".xml",  "xml" ), (".tag",  "tag" )):
    #         if A.outfile.endswith(override):
    #             A.output_type = alttype
    #
    # if A.outfile == "-":
    #     outfp = sys.stdout
    #     if outfp.encoding is not None:
    #         # Why ignore outfp.encoding? :-/ stupid cathal?
    #         A.codec = 'utf-8'
    # else:
    #     outfp = open(A.outfile, "wb")


    convert_pdf_to_txt(files=[], outfile='-',
                _py2_no_more_posargs=None,  # Bloody Python2 needs a shim
                no_laparams=False, all_texts=None, detect_vertical=None, # LAParams
                word_margin=None, char_margin=None, line_margin=None, boxes_flow=None, # LAParams
                output_type='text', codec='utf-8', strip_control=False,
                maxpages=0, page_numbers=None, password="", scale=1.0, rotation=0,
                layoutmode='normal', output_dir=None, debug=False,
                disable_caching=False)
    # convert_pdf_to_txt(**vars(A))
    ## Test Code
    #outfp = extract_text(**vars(A))
    #outfp.close()
    return 0

#======================================================================================================================
if __name__ == '__main__':
    sys.exit(main())
