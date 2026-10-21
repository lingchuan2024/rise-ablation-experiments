import json
import unittest

from rise.agent import ToolCallRecord, _tool_call_to_trace, _tool_returned_error


class V11TraceTests(unittest.TestCase):
    def test_structured_tool_and_shell_failures_are_counted_as_errors(self):
        self.assertEqual(_tool_returned_error('{"status":"error","error":"unknown source"}'), 'tool_returned_error')
        self.assertEqual(_tool_returned_error('{"status":"ok","command_status":"error"}'), 'tool_returned_error')
        self.assertEqual(_tool_returned_error('{"status":"ok","text":"error is a noun"}'), '')

    def trace(self, name, args, payload):
        result = json.dumps(payload)
        call = ToolCallRecord(name=name, args=args, result=result, result_chars=len(result))
        return _tool_call_to_trace(call, {'example/source.txt': 'doc-7'})

    def test_dci_preview_preserves_parent_docid_and_view_metadata(self):
        row = self.trace('dci', {'goal_query': 'actor debut'}, {
            'status': 'ok', 'view_id': 'V-1', 'space_version': 2,
            'goal_query': 'actor debut', 'top_k': 100, 'cache_hit': False,
            'items': [{'file_path': 'P-1.txt', 'source_file_path': 'example/source.txt'}]})
        self.assertEqual(row['result_docids'], ['doc-7'])
        self.assertEqual(row['space_metadata']['goal_query'], 'actor debut')
        self.assertEqual(row['result_unresolved_paths'], [])

    def test_dci_shell_surfaced_passage_maps_to_parent(self):
        row = self.trace('dci', {'command': 'rg debut *.txt'}, {
            'status': 'ok', 'command_executed': True, 'items': [],
            'surfaced_passages': [{'source_file_path': 'example/source.txt'}]})
        self.assertEqual(row['result_docids'], ['doc-7'])

    def test_passage_read_and_source_expansion_map_to_same_parent(self):
        read = self.trace('read', {'view_id': 'V-1', 'passage_id': 'P-1'}, {
            'status': 'ok', 'passage': {'source_file_path': 'example/source.txt'}})
        source = self.trace('expand_source', {'source_id': 'SRC-1'}, {
            'status': 'ok', 'source': {'file_path': 'example/source.txt'},
            'char_start': 100, 'char_end': 300})
        self.assertEqual(read['result_docids'], source['result_docids'])
        self.assertEqual(source['space_metadata']['char_end'], 300)


if __name__ == '__main__':
    unittest.main()
