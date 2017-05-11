from itertools import chain

from lxml import html

from data.language import Sample


def ParseHtml(story, corpus): # from
    """Parses the HTML of a news story.
    Args:
      story: The raw Story to be parsed.
      corpus: Either 'cnn' or 'dailymail'.
    Returns:
      A Story containing URL, paragraphs and highlights.
    """

    parser = html.HTMLParser(encoding='utf-8')
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
    url = './cnn/00a2aef1e18d125960da51e167a3d22ed8416c09.html'
    with open(url) as f:
        story_html = f.read()
    corpus = 'cnn'
    a = ParseHtml(story_html, corpus)
    sample = Sample(headline='', body=a)
    print(a)

