import json
from itertools import chain

import sys
from math import ceil

from bs4 import BeautifulSoup

from lxml import html
from os import listdir, path

import chardet


def read_cnn_data(input_dir, output_dir, batch_size=5000):
    files = [path.join(input_dir, f) for f in listdir(cnn_dir)]
    print('Found', len(files), 'files...')

    def chunks(l, n):
        """Yield successive n-sized chunks from l."""
        """from http://stackoverflow.com/a/312464"""
        for i in range(0, len(l), n):
            yield l[i:i + n]

    num_batches = ceil(len(files) / batch_size)
    error_docs = 0
    for batch_i, batch in enumerate(chunks(files, batch_size)):
        print('Processing batch {}/{}'.format(batch_i+1, num_batches))
        articles = []
        for file_i, file in enumerate(batch):
            if file_i and not file_i % 1000:
                print('Processing file {}/{} of batch'.format(file_i, len(batch)))
            try:
                articles.append(read_file_guessing_charsets(file, ['utf-8', 'ISO-8859-1', None]))
            except Exception as e:
                print('Error processing file {}:{} at {}'.format(batch_i, file_i, file))
                print(e)
                error_docs += 1

        with open(path.join(output_dir, 'cnn-{}.json'.format(batch_i+1)), 'w') as fp:
            json.dump(articles, fp, indent=4)

    print('Successful docs: {}, error docs: {}'.format(len(files) - error_docs, error_docs))


def read_file_guessing_charsets(file, charsets):
    """
    Try to read a file guessing at the charsets. This is faster than trying to determine the charset for each file,
    :param file: file to parse
    :param charsets: list of charsets to guess in likelihood order. pass None at the end of the list to use chardet if
    all else fails
    :return: parsed file
    """
    for charset in charsets:
        try:
            return read_file(file, charset)
        except Exception as e:
            continue
    raise e


cnn_len_title = len(' - CNN.com')
cnn_len_body = len('(CNN) -- ')


def read_file(file, charset=None):
    with open(file, 'rb') as f:
        story_bytes = f.read()
        encoding = chardet.detect(story_bytes)['encoding'] if not charset else charset
        if not charset:
            print(encoding)
        soup = BeautifulSoup(story_bytes, 'html.parser', from_encoding=encoding)
        title = soup.find('title').text
        if title.endswith(' - CNN.com'):
            title = title[:-cnn_len_title]
        body = ParseHtml(story_bytes.decode(encoding), 'cnn')
        if body.startswith('(CNN) -- '):
            body = body[cnn_len_body:]
        return dict(headline=title, body=body)


def ParseHtml(story, corpus, encoding='utf-8'):
    """Adapted from https://github.com/deepmind/rc-data"""
    """Parses the HTML of a news story.
    Args:
      story: The raw HTML to be parsed.
      corpus: Either 'cnn' or 'dailymail'.
    Returns:
      A Story containing URL, paragraphs and highlights.
    """
    parser = html.HTMLParser(encoding=encoding)
    tree = html.document_fromstring(story, parser=parser)

    # Elements to delete.
    delete_selectors = {
        'cnn': [
            '//blockquote[contains(@class, "twitter-tweet")]',
            '//blockquote[contains(@class, "instagram-media")]'
        ],
        'dailymail': [
            '//blockquote[contains(@class, "twitter-tweet")]',
            '//blockquote[contains(@class, "instagram-media")]'
        ]
    }

    # Paragraph exclusions: ads, links, bylines, comments
    cnn_exclude = (
        'not(ancestor::*[contains(@class, "metadata")])'
        ' and not(ancestor::*[contains(@class, "pullquote")])'
        ' and not(ancestor::*[contains(@class, "SandboxRoot")])'
        ' and not(ancestor::*[contains(@class, "twitter-tweet")])'
        ' and not(ancestor::div[contains(@class, "cnnStoryElementBox")])'
        ' and not(contains(@class, "cnnTopics"))'
        ' and not(descendant::*[starts-with(text(), "Read:")])'
        ' and not(descendant::*[starts-with(text(), "READ:")])'
        ' and not(descendant::*[starts-with(text(), "Join us at")])'
        ' and not(descendant::*[starts-with(text(), "Join us on")])'
        ' and not(descendant::*[starts-with(text(), "Read CNNOpinion")])'
        ' and not(descendant::*[contains(text(), "@CNNOpinion")])'
        ' and not(descendant-or-self::*[starts-with(text(), "Follow us")])'
        ' and not(descendant::*[starts-with(text(), "MORE:")])'
        ' and not(descendant::*[starts-with(text(), "SPOILER ALERT:")])')

    dm_exclude = (
        'not(ancestor::*[contains(@id,"reader-comments")])'
        ' and not(contains(@class, "byline-plain"))'
        ' and not(contains(@class, "byline-section"))'
        ' and not(contains(@class, "count-number"))'
        ' and not(contains(@class, "count-text"))'
        ' and not(contains(@class, "video-item-title"))'
        ' and not(ancestor::*[contains(@class, "column-content")])'
        ' and not(ancestor::iframe)')

    paragraph_selectors = {
        'cnn': [
            '//div[contains(@class, "cnnContentContainer")]//p[%s]' % cnn_exclude,
            '//div[contains(@class, "l-container")]//p[%s]' % cnn_exclude,
            '//div[contains(@class, "cnn_strycntntlft")]//p[%s]' % cnn_exclude
        ],
        'dailymail': [
            '//div[contains(@class, "article-text")]//p[%s]' % dm_exclude
        ]
    }

    # Highlight exclusions.
    he = (
        'not(contains(@class, "cnnHiliteHeader"))'
        ' and not(descendant::*[starts-with(text(), "Next Article in")])')
    highlight_selectors = {
        'cnn': [
            '//*[contains(@class, "el__storyhighlights__list")]//li[%s]' % he,
            '//*[contains(@class, "cnnStryHghLght")]//li[%s]' % he,
            '//*[@id="cnnHeaderRightCol"]//li[%s]' % he
        ],
        'dailymail': [
            '//h1/following-sibling::ul//li'
        ]
    }

    def ExtractText(selector):
        """Extracts a list of paragraphs given a XPath selector.
        Args:
          selector: A XPath selector to find the paragraphs.
        Returns:
          A list of raw text paragraphs with leading and trailing whitespace.
        """

        xpaths = map(tree.xpath, selector)
        elements = list(chain.from_iterable(xpaths))
        paragraphs = [e.text_content() for e in elements]

        # Remove editorial notes, etc.
        if corpus == 'cnn' and len(paragraphs) >= 2 and '(CNN)' in paragraphs[1]:
            paragraphs.pop(0)

        paragraphs = map(str.strip, paragraphs)
        paragraphs = [s for s in paragraphs if s and not str.isspace(s)]

        return paragraphs

    for selector in delete_selectors[corpus]:
        for bad in tree.xpath(selector):
            bad.getparent().remove(bad)

    paragraphs = ExtractText(paragraph_selectors[corpus])

    return '\n'.join(paragraphs)


if __name__ == '__main__':
    cnn_dir = sys.argv[1]
    output_dir = sys.argv[2]
    print('Input dir {}, output dir {}'.format(cnn_dir, output_dir))
    read_cnn_data(cnn_dir, output_dir)
