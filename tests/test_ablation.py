import unittest
from src.ablation import remove_comments
class CommentTests(unittest.TestCase):
    def test_literals_and_separation(self):
        source='char *s="https://host/*x*/"; char c=\'/\'; int/**/x; // remove\nreturn x;'
        result=remove_comments(source)
        self.assertIn('"https://host/*x*/"',result)
        self.assertIn("'/'",result)
        self.assertIn('int x',result)
        self.assertNotIn('remove',result)
    def test_escaped_quotes(self):
        self.assertEqual(remove_comments(r'"a\"//b" /*x*/'),r'"a\"//b"  ')
    def test_line_splicing(self):
        self.assertEqual(remove_comments('// a\\\n continued\nint x;'),' \nint x;')
    def test_multiline(self):
        self.assertEqual(remove_comments('a/*hello\nthere*/b'),'a \nb')
if __name__=='__main__': unittest.main()
