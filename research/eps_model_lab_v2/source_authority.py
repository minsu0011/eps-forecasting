"""Filing-local iXBRL authority with explicit contexts, scales and quarantine."""
from decimal import Decimal, InvalidOperation
import re
from pathlib import Path
import sys
import time
import urllib.request
import urllib.error
from lxml import html

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from research.eps_model_lab_v2.common import RUN, utcnow, sha, save_json, require_output, check_stop


def local(tag):
    return str(tag).split('}')[-1].split(':')[-1].lower()


def children(node, name):
    return [c for c in node.iter() if local(c.tag) == name.lower()]


def parse_ixbrl(payload, accession, ticker, cik, source_url):
    root = html.fromstring(payload)
    contexts, units = {}, {}
    for node in root.iter():
        name = local(node.tag)
        if name == 'context':
            def value(key):
                found = children(node, key)
                return ''.join(found[0].itertext()).strip() if len(found) == 1 else None
            identifier = value('identifier')
            contexts[node.get('id')] = {'start':value('startdate'), 'end':value('enddate') or value('instant'),
                'instant':bool(children(node,'instant')), 'identifier':identifier,
                'dimensions_present':bool(children(node,'explicitmember') or children(node,'typedmember') or children(node,'segment') or children(node,'scenario'))}
        elif name == 'unit':
            nums = children(node,'unitnumerator')
            dens = children(node,'unitdenominator')
            def measures(parent):
                return [local(''.join(c.itertext()).strip()) for c in children(parent,'measure')]
            units[node.get('id')] = ('/'.join(['*'.join(measures(nums[0])), '*'.join(measures(dens[0]))])
                if len(nums) == len(dens) == 1 else '*'.join(measures(node)))
    records = []
    allowed_formats = {'','num-dot-decimal','numdotdecimal','num-comma-dot','numcommadot','numdash','num-dash','zerodash'}
    for node in root.iter():
        if local(node.tag) != 'nonfraction':
            continue
        tag = node.get('name','')
        context = contexts.get(node.get('contextref'))
        if context is None:
            continue
        text = ''.join(node.itertext()).strip()
        record = {'ticker':ticker, 'cik':int(cik), 'accession':accession, 'concept':tag,
            'context_id':node.get('contextref'), 'unitRef':node.get('unitref'),
            'unit':units.get(node.get('unitref')), 'scale':node.get('scale','0'),
            'decimals':node.get('decimals'), 'source_value':text, 'source_url':source_url,
            'canonical_value':None, 'status':'QUARANTINED', **context}
        try:
            if record['unit'] is None:
                raise ValueError('Missing or unresolved unitRef')
            if context['dimensions_present']:
                raise ValueError('Dimensional context is not consolidated entity authority')
            if int(context['identifier']) != int(cik):
                raise ValueError('Context entity does not match CIK')
            fmt = local(node.get('format',''))
            if fmt not in allowed_formats:
                raise ValueError('Unsupported numeric transformation: '+fmt)
            if any(local(k) == 'nil' and v.lower() in ('true','1') for k,v in node.attrib.items()):
                raise ValueError('Nil fact')
            if children(node,'exclude') or node.get('continuedat'):
                raise ValueError('Exclude/continuation needs explicit parser support')
            compact = text.replace('\u00a0','').replace(',','').replace(' ','').strip()
            if compact in ('-','—','–') and fmt in ('numdash','num-dash','zerodash'):
                compact = '0'
            if not re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)', compact):
                raise ValueError('Unrecognized numeric lexical value')
            value = Decimal(compact)*Decimal(10)**int(record['scale'])
            if node.get('sign') == '-':
                value = -value
            if not value.is_finite():
                raise ValueError('Nonfinite canonical value')
            record.update(canonical_value=float(value), status='VERIFIED',
                authority='ORIGINAL_FILING_IXBRL_EXACT_CONTEXT_UNITREF_SCALE')
        except (ValueError, TypeError, InvalidOperation, OverflowError) as exc:
            record['quarantine_reason'] = str(exc)
        records.append(record)
    return records


def fetch_original(url, key):
    check_stop()
    path = require_output(RUN/'data/source_filings'/f'{key}.htm')
    receipt = RUN/'data/source_filings'/f'{key}.receipt.json'
    if receipt.exists():
        raise FileExistsError('Source attempt already recorded; do not silently retry an identity')
    result = {'created_utc':utcnow(), 'url':url, 'path':str(path), 'key':key,
        'user_agent':'EPSModelLabV2 academic-style local research', 'no_authentication_or_bypass':True}
    try:
        req = urllib.request.Request(url, headers={'User-Agent':result['user_agent'], 'Accept':'text/html,application/xhtml+xml'})
        with urllib.request.urlopen(req, timeout=35) as response:
            payload = response.read(24*1024*1024+1)
            if len(payload)>24*1024*1024:
                raise ValueError('Filing exceeds explicit 24 MiB fetch limit')
            if b'<html' not in payload[:5000].lower():
                raise ValueError('Unexpected non-HTML body')
            path.write_bytes(payload)
            result.update(status='DOWNLOADED', bytes=len(payload), sha256=sha(path), http_status=response.status)
    except Exception as exc:
        result.update(status='SOURCE_ACCESS_UNAVAILABLE_QUARANTINE', error=f'{type(exc).__name__}: {exc}')
    save_json(receipt, result, immutable=True)
    return result


def probe():
    items = [
        ('MCD_2023','https://www.sec.gov/Archives/edgar/data/63908/000006390824000072/mcd-20231231.htm','MCD',63908,'0000063908-24-000072'),
        ('AMZN_2025Q1','https://www.sec.gov/Archives/edgar/data/1018724/000101872425000036/amzn-20250331.htm','AMZN',1018724,'0001018724-25-000036')]
    rows = []
    for key,url,ticker,cik,accn in items:
        result = fetch_original(url,key)
        if result['status'] == 'DOWNLOADED':
            facts = parse_ixbrl(Path(result['path']).read_bytes(),accn,ticker,cik,url)
            save_json(RUN/'data/source_filings'/f'{key}.facts.json',facts,immutable=True)
            result['parsed_facts'] = len(facts)
        rows.append(result)
        print(key,result['status'],result.get('parsed_facts'),result.get('error'),flush=True)
        time.sleep(.6)
    save_json(RUN/'audit/ORIGINAL_SOURCE_ACCESS_PROBE.json', {'attempts':rows,'created_utc':utcnow()},immutable=True)


if __name__ == '__main__':
    probe()
