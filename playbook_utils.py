"""
==============================================================================
playbook.py
==============================================================================

This file contains functions for parsing and manipulating the playbook.

"""
import json
import re
from utils import get_section_slug

def parse_playbook_line(line):
    """Parse a single playbook line to extract components"""
    # Pattern: [id] helpful=X harmful=Y :: content
    pattern = r'\[([^\]]+)\]\s*helpful=(\d+)\s*harmful=(\d+)\s*::\s*(.*)'
    match = re.match(pattern, line.strip())
    
    if match:
        return {
            'id': match.group(1),
            'helpful': int(match.group(2)),
            'harmful': int(match.group(3)),
            'content': match.group(4),
            'raw_line': line
        }
    return None

def get_next_global_id(playbook_text):
    """Extract highest global ID and return next one"""
    max_id = 0
    lines = playbook_text.strip().split('\n')
    
    for line in lines:
        parsed = parse_playbook_line(line)
        if parsed:
            # Extract numeric part from ID
            id_match = re.search(r'-(\d+)$', parsed['id'])
            if id_match:
                num = int(id_match.group(1))
                max_id = max(max_id, num)
    
    return max_id + 1


def format_playbook_line(bullet_id, helpful, harmful, content):
    """Format a bullet into playbook line format"""
    return f"[{bullet_id}] helpful={helpful} harmful={harmful} :: {content}"

def update_bullet_counts(playbook_text, bullet_tags):
    """Update helpful/harmful counts based on tags (Counter layer)"""
    lines = playbook_text.strip().split('\n')
    updated_lines = []
    
    # Create tag lookup - handle both old and new formats
    tag_map = {}
    if isinstance(bullet_tags, list) and len(bullet_tags) > 0:
        for tag in bullet_tags:
            if isinstance(tag, dict):
                # Handle both 'id' and 'bullet' keys for backwards compatibility
                bullet_id = tag.get('id') or tag.get('bullet', '')
                tag_value = tag.get('tag', 'neutral')
                if bullet_id:
                    tag_map[bullet_id] = tag_value
    
    if not tag_map:
        print("Warning: No valid bullet tags found to update counts")
        return playbook_text
    
    for line in lines:
        if line.strip().startswith('#') or not line.strip():
            # Preserve section headers and empty lines
            updated_lines.append(line)
            continue
            
        parsed = parse_playbook_line(line)
        if parsed and parsed['id'] in tag_map:
            tag = tag_map[parsed['id']]
            if tag == 'helpful':
                parsed['helpful'] += 1
            elif tag == 'harmful':
                parsed['harmful'] += 1
            # neutral: no change
            
            # Reconstruct line with updated counts
            new_line = format_playbook_line(
                parsed['id'], parsed['helpful'], parsed['harmful'], parsed['content']
            )
            updated_lines.append(new_line)
        else:
            updated_lines.append(line)
    
    return '\n'.join(updated_lines)


def apply_curator_operations(playbook_text, operations, next_id):
    """
    Apply curator operations to playbook
    
    TODO: Future Operations (not implemented yet)
    - UPDATE: Rewrite existing bullets to be more accurate or comprehensive
    - MERGE: Combine related bullets into stronger ones  
    - CREATE_META: Add high-level strategy sections
    - DELETE: Remove outdated or incorrect bullets (if needed)
    """
    lines = playbook_text.strip().split('\n')

    def normalize_section(value):
        return value.strip().lower().replace(' ', '_').replace('&', 'and')

    section_by_id, known_sections = {}, set()
    current_section = 'general'
    for line in lines:
        if line.strip().startswith('##'):
            current_section = normalize_section(line.strip()[2:])
            known_sections.add(current_section)
        else:
            parsed = parse_playbook_line(line)
            if parsed:
                section_by_id[parsed['id']] = current_section

    updates, deletes, additions, created_sections = {}, set(), [], []
    active_ids = set(section_by_id)
    for op in operations:
        if not isinstance(op, dict):
            continue
        op_type = op.get('type', '').upper()
        if op_type == 'CREATE_META':
            title = op.get('title') or op.get('section_name') or op.get('section')
            if title:
                section = normalize_section(op.get('section_id') or title)
                if section not in known_sections:
                    known_sections.add(section)
                    created_sections.append((section, title))
            continue
        if op_type == 'ADD':
            content = op.get('content', '').strip()
            if not content:
                continue
            section = normalize_section(op.get('section', 'others'))
            if section not in known_sections:
                section = 'others'
            new_id = f"{get_section_slug(section)}-{next_id:05d}"
            next_id += 1
            additions.append((section, format_playbook_line(new_id, 0, 0, content)))
            print(f"  Added bullet {new_id} to section {section}")
            continue
        if op_type == 'UPDATE':
            target_id, content = op.get('target_id', ''), op.get('content', '').strip()
            if target_id in active_ids and content:
                updates[target_id] = content
                print(f"  Updated bullet {target_id}")
            else:
                reason = "unknown target" if target_id not in active_ids else "empty content"
                known_preview = ", ".join(sorted(active_ids)[:12]) or "(no active bullets)"
                print(
                    f"  Skipped UPDATE: {reason} ({target_id}). "
                    f"Known IDs: {known_preview}"
                )
                op["_execution_status"] = "skipped"
                op["_execution_reason"] = reason
            continue
        if op_type == 'DELETE':
            target_id = op.get('target_id', '')
            if target_id in active_ids and op.get('reason', '').strip():
                deletes.add(target_id)
                print(f"  Deleted bullet {target_id}")
            else:
                reason = "unknown target" if target_id not in active_ids else "missing reason"
                print(f"  Skipped DELETE: {reason} ({target_id})")
                op["_execution_status"] = "skipped"
                op["_execution_reason"] = reason
            continue
        if op_type == 'MERGE':
            source_ids = list(dict.fromkeys(op.get('source_ids', [])))
            source_ids = [bullet_id for bullet_id in source_ids if bullet_id in active_ids and bullet_id not in deletes]
            content = op.get('content', '').strip()
            if len(source_ids) < 2 or not content:
                print("  Skipped MERGE: need two valid source IDs and merged content")
                continue
            section = normalize_section(op.get('section') or section_by_id[source_ids[0]])
            if section not in known_sections:
                section = section_by_id[source_ids[0]]
            source_stats = []
            for line in lines:
                parsed = parse_playbook_line(line)
                if parsed and parsed['id'] in source_ids:
                    source_stats.append(parsed)
            new_id = f"{get_section_slug(section)}-{next_id:05d}"
            next_id += 1
            additions.append((section, format_playbook_line(
                new_id,
                sum(item['helpful'] for item in source_stats),
                sum(item['harmful'] for item in source_stats),
                content,
            )))
            deletes.update(source_ids)
            print(f"  Merged {', '.join(source_ids)} into {new_id}")

    pending = {}
    for section, line in additions:
        pending.setdefault(section, []).append(line)
    final_lines, current_section = [], 'general'
    for line in lines:
        if line.strip().startswith('##'):
            if current_section in pending:
                final_lines.extend(pending.pop(current_section))
            current_section = normalize_section(line.strip()[2:])
            final_lines.append(line)
            continue
        parsed = parse_playbook_line(line)
        if parsed and parsed['id'] in deletes:
            continue
        if parsed and parsed['id'] in updates:
            final_lines.append(format_playbook_line(
                parsed['id'], parsed['helpful'], parsed['harmful'], updates[parsed['id']]
            ))
        else:
            final_lines.append(line)
    if current_section in pending:
        final_lines.extend(pending.pop(current_section))
    for section, title in created_sections:
        final_lines.extend(['', f"## {title.upper()}"])
        final_lines.extend(pending.pop(section, []))
    if pending:
        if 'others' not in known_sections:
            final_lines.extend(['', '## OTHERS'])
        for section_lines in pending.values():
            final_lines.extend(section_lines)
    return '\n'.join(final_lines), next_id

def get_playbook_stats(playbook_text):
    """Generate statistics about the playbook"""
    lines = playbook_text.strip().split('\n')
    stats = {
        'total_bullets': 0,
        'high_performing': 0,  # helpful > 5, harmful < 2
        'problematic': 0,      # harmful >= helpful
        'unused': 0,           # helpful + harmful = 0
        'by_section': {}
    }
    
    current_section = 'general'
    
    for line in lines:
        if line.strip().startswith('##'):
            current_section = line.strip()[2:].strip()
            continue
            
        parsed = parse_playbook_line(line)
        if parsed:
            stats['total_bullets'] += 1
            
            if parsed['helpful'] > 5 and parsed['harmful'] < 2:
                stats['high_performing'] += 1
            elif parsed['harmful'] >= parsed['helpful'] and parsed['harmful'] > 0:
                stats['problematic'] += 1
            elif parsed['helpful'] + parsed['harmful'] == 0:
                stats['unused'] += 1
            
            if current_section not in stats['by_section']:
                stats['by_section'][current_section] = {'count': 0, 'helpful': 0, 'harmful': 0}
            
            stats['by_section'][current_section]['count'] += 1
            stats['by_section'][current_section]['helpful'] += parsed['helpful']
            stats['by_section'][current_section]['harmful'] += parsed['harmful']
    
    return stats

def extract_json_from_text(text, json_key=None):
    """Extract JSON object from text, handling various formats"""
    try:
        # First, try to parse the entire response as JSON (JSON mode)
        try:
            result = json.loads(text.strip())
            return result
        except json.JSONDecodeError:
            pass
        
        # Fallback: Look for ```json blocks
        json_pattern = r'```json\s*(.*?)\s*```'
        matches = re.findall(json_pattern, text, re.DOTALL | re.IGNORECASE)
        
        if matches:
            # Try each match until we find valid JSON
            for match in matches:
                try:
                    json_str = match.strip()
                    result = json.loads(json_str)
                    return result
                except json.JSONDecodeError:
                    continue
        
        # Improved JSON extraction using balanced brace counting
        # This handles deeply nested structures better
        def find_json_objects(text):
            """Find JSON objects using balanced brace counting"""
            json_objects = []
            i = 0
            while i < len(text):
                if text[i] == '{':
                    # Found start of potential JSON object
                    brace_count = 1
                    start = i
                    i += 1
                    
                    while i < len(text) and brace_count > 0:
                        if text[i] == '{':
                            brace_count += 1
                        elif text[i] == '}':
                            brace_count -= 1
                        elif text[i] == '"':
                            # Handle quoted strings to avoid counting braces inside strings
                            i += 1
                            while i < len(text) and text[i] != '"':
                                if text[i] == '\\':
                                    i += 1  # Skip escaped character
                                i += 1
                        i += 1
                    
                    if brace_count == 0:
                        # Found complete JSON object
                        json_candidate = text[start:i]
                        json_objects.append(json_candidate)
                else:
                    i += 1
            
            return json_objects
        
        # Find all potential JSON objects
        json_objects = find_json_objects(text)
        
        for json_str in json_objects:
            try:
                result = json.loads(json_str)
                return result
            except json.JSONDecodeError:
                continue

        def find_object_end(text, start_index):
            """Return the end index for a balanced JSON object starting at start_index."""
            depth = 0
            in_string = False
            escape = False

            for index in range(start_index, len(text)):
                char = text[index]

                if in_string:
                    if escape:
                        escape = False
                    elif char == '\\':
                        escape = True
                    elif char == '"':
                        in_string = False
                    continue

                if char == '"':
                    in_string = True
                elif char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0:
                        return index + 1

            return None

        def recover_array_items(text, key):
            """Recover complete JSON objects from a possibly truncated array field."""
            if not key:
                return None

            match = re.search(rf'"{re.escape(key)}"\s*:\s*\[', text)
            if not match:
                return None

            items = []
            index = match.end()

            while index < len(text):
                while index < len(text) and text[index] in ' \t\r\n,':
                    index += 1

                if index >= len(text) or text[index] == ']':
                    break

                if text[index] != '{':
                    index += 1
                    continue

                object_end = find_object_end(text, index)
                if object_end is None:
                    break

                candidate = text[index:object_end]
                try:
                    items.append(json.loads(candidate))
                except json.JSONDecodeError:
                    pass

                index = object_end

            return items or None

        recovered_items = recover_array_items(text, json_key)
        if recovered_items is not None:
            return {json_key: recovered_items}
                
    except Exception as e:
        print(f"Failed to extract JSON: {e}")
        if len(text) > 500:
            print(f"Raw content preview:\n{text[:500]}...")
        else:
            print(f"Raw content:\n{text}")
        
    return None

def extract_playbook_bullets(playbook_text, bullet_ids):
    """
    Extract specific bullet points from playbook based on bullet_ids.
    
    Args:
        playbook_text (str): The full playbook text
        bullet_ids (list): List of bullet IDs to extract
    
    Returns:
        str: Formatted playbook content containing only the specified bullets
    """
    if not bullet_ids:
        return "(No bullets used by generator)"
    
    lines = playbook_text.strip().split('\n')
    found_bullets = []
    
    for line in lines:
        if line.strip():  # Skip empty lines
            parsed = parse_playbook_line(line)
            if parsed and parsed['id'] in bullet_ids:
                found_bullets.append({
                    'id': parsed['id'],
                    'content': parsed['content'],
                    'helpful': parsed['helpful'],
                    'harmful': parsed['harmful']
                })
    
    if not found_bullets:
        return "(Generator referenced bullet IDs but none were found in playbook)"
    
    # Format the bullets for reflector input
    formatted_bullets = []
    for bullet in found_bullets:
        formatted_bullets.append(f"[{bullet['id']}] helpful={bullet['helpful']} harmful={bullet['harmful']} :: {bullet['content']}")
    
    return '\n'.join(formatted_bullets)
