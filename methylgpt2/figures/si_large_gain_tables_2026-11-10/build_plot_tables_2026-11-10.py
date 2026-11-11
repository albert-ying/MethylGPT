"""Select and reshape released aggregate statistics without statistical arithmetic.

Run from the repository root. All writes stay beside this script. No draws,
participant records, external services, or statistical recalculation are used.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
TABLES = {}
TRACE = {}
CACHE = {}
OMISSIONS = []


def locate(folder, name):
    matches = list((ROOT / 'artifacts').glob(f'large_gain_*/{folder}/{name}'))
    matches = [p for p in matches if '_private' not in str(p)]
    if len(matches) != 1:
        raise FileNotFoundError(f'Cannot proceed - required data file not found: artifacts/large_gain_*/{folder}/{name}')
    path = matches[0]
    if any(s in path.name for s in ('bootstrap_draw', 'per_draw', 'private_', 'pairs_frozen')):
        raise ValueError(f'Prohibited input: {path}')
    return str(path.relative_to(ROOT))


def read(path):
    if path not in CACHE:
        try:
            raw = (ROOT / path).read_text()
            CACHE[path] = json.loads(raw) if path.endswith('.json') else list(csv.DictReader(io.StringIO(raw)))
        except OSError as exc:
            raise FileNotFoundError(f'Cannot proceed - required data file not found: {path}') from exc
    return CACHE[path]


def select(path, **filters):
    rows = [r for r in read(path) if all(r.get(k) == str(v) for k, v in filters.items())]
    if len(rows) != 1:
        raise ValueError(f'Expected one aggregate row: {path} {filters}; found {len(rows)}')
    return path, filters, rows[0]


def nested(obj, keys):
    for key in keys:
        obj = obj[key]
    return obj


def ref(row, key):
    path, filters, record = row
    keys = list(key) if isinstance(key, tuple) else [key]
    try:
        value = nested(record, keys)
    except KeyError as exc:
        raise KeyError(f'Cannot proceed - required data file not found: {path} column {key}') from exc
    return value, {'file': path, 'filter': filters, 'column': keys}


def literal(value):
    return value, None


def put(name, fields):
    row, provenance = {}, {}
    for key, (value, origin) in fields.items():
        row[key] = value
        if origin:
            provenance[key] = origin
    if row.get('suppressed') == 'True':
        raise ValueError('A suppressed source cannot produce a plotted mark')
    if row.get('value') in ('', None):
        origin = provenance['value']
        OMISSIONS.append(f"Cannot proceed - required data file not found: {origin['file']} {origin['filter']} column {origin['column']}")
        return
    if not math.isfinite(float(row['value'])):
        raise ValueError(f'Non-finite plotted value: {name} {row}')
    for bound in ('ci95_low', 'ci95_high'):
        if row[bound] not in ('', None) and not math.isfinite(float(row[bound])):
            raise ValueError(f'Non-finite interval: {name}')
    origin = provenance['value']
    row['source_file'] = origin['file']
    row['source_row_filter'] = json.dumps(origin['filter'], ensure_ascii=False, sort_keys=True)
    row['source_value_column'] = json.dumps(origin['column'], ensure_ascii=False)
    for bound in ('ci95_low', 'ci95_high'):
        row[f'source_{bound}_column'] = json.dumps(provenance[bound]['column']) if bound in provenance else ''
    TABLES.setdefault(name, []).append(row)
    TRACE.setdefault(name, []).append(provenance)


def mark(name, source, value, label, metric, unit, low=None, high=None, **extra):
    fields = {
        'label': literal(label), 'metric': literal(metric), 'unit': literal(unit),
        'value': ref(source, value),
        'ci95_low': ref(source, low) if low else literal(''),
        'ci95_high': ref(source, high) if high else literal(''),
    }
    fields.update(extra)
    fields.setdefault('note', literal(''))
    put(name, fields)


J_RULE = locate('J', 'J_rule_outcome_*.csv')
J_SUM = locate('J', 'J_summary_*.csv')
J_MASK = locate('J', 'J_masks_*.csv')
R_RULE = locate('R', 'R_rule_outcome_*.csv')
R_CLOCK = locate('R', 'R_clock_contrasts_*.csv')
M_RULE = locate('M', 'M_rule_outcome_*.csv')
M_SUM = locate('M', 'M_summary_*.csv')
M_INPUT = locate('M', 'M_inputs_*.csv')
O_SUM = locate('O', 'O_summary_*.csv')
O_MET = locate('O', 'O_metrics_*.csv')
S_HR = locate('S', 'S_hr_contrasts_*.csv')
V_C = locate('V', 'V_cindex_mean_contrasts_*.csv')
V_H = locate('V', 'V_hr_contrasts_*.csv')
D_RULE = locate('D', 'D_rule_outcome.csv')
D2_END = locate('D2', 'D2_per_endpoint.csv')
D2_RULE = locate('D2', 'D2_rule_outcome.csv')
N_RULE = locate('N', 'N_rule_outcome.csv')
N_APP = locate('N', 'N_applications.csv')
U_DISEASE = locate('U', 'U_disease_contrasts_*.csv')
U_MEAN = locate('U', 'U_contrasts_*.csv')
K_DISEASE = locate('K', 'K_disease_table_*.csv')
K_RULE = locate('K', 'K_rule_outcome_*.csv')
C_PER = locate('C', 'per_comparison_metrics.csv')
C_SUM = locate('C', 'summary_metrics.csv')
G_SUM = locate('G', 'summary_heldout_*.csv')
G_JSON = locate('G', 'results_*.json')
G_PER = locate('G', 'per_dataset_auc_heldout_*.csv')
G_DATA = locate('G', 'datasets_G_*.csv')
G_MASK = locate('G', 'masks_heldout_*.csv')

METHODS = {
    'mgpt2_combined': 'MethylGPT',
    'mgpt2_combined_depthaware': 'MethylGPT with read-depth weighting',
    'grimage2_mean': 'GrimAge2 (mean filling)',
    'grimage2_knn': 'GrimAge2 (nearest-neighbour filling)',
    'dunedinpace_mean': 'DunedinPACE (mean filling)',
    'dunedinpace_knn': 'DunedinPACE (nearest-neighbour filling)',
    'raw_cpg_gpu_mean': 'Raw CpGs (dense fit, mean filling)',
    'raw_cpg_gpu_knn': 'Raw CpGs (dense fit, nearest-neighbour filling)',
    'raw_cpg_v12_mean': 'Raw CpGs (elastic-net fit, mean filling)',
    'raw_cpg_v12_knn': 'Raw CpGs (elastic-net fit, nearest-neighbour filling)',
    'raw_cpg_v12_knndenoise': 'Raw CpGs (elastic-net fit, nearest-neighbour denoising)',
}
SCENARIOS = {
    'full': 'Complete profiles', 'hm27': '27K-array CpGs',
    'random_90': 'Random 10% of CpGs retained',
    'random_95': 'Random 5% of CpGs retained',
    'seq_GSE194154': 'RRBS whole-blood coverage (GSE194154)',
    'seq_GSE282512': 'RRBS plasma coverage (GSE282512)',
    'seq_GSE140730': 'WGBS cord-blood coverage (GSE140730)',
    **{f'gauss_{s}': f'Gaussian noise SD {s}' for s in ('0.02', '0.05', '0.10')},
    **{f'depth_{d}': f'Mean read depth {d}' for d in ('30', '10', '5', '2', '1')},
}
NOISE_ORDER = ['full', 'gauss_0.02', 'gauss_0.05', 'gauss_0.10', 'depth_30', 'depth_10', 'depth_5', 'depth_2', 'depth_1']
ENDPOINTS = {
    'Endocrine_and_Metabolic_Diseases': 'Type 2 diabetes',
    'Autoimmune_Diseases': 'Autoimmune disease', 'Cancers': 'Cancer',
    'Cardiovascular_Diseases': 'Cardiovascular disease', 'Kidney_Diseases': 'Kidney disease',
    'Neurological_Diseases': 'Neurological disease', 'Respiratory_Diseases': 'Respiratory disease',
    'date_of_death': 'Death',
}
COHORT = select(N_RULE, scenario='depth_10')


def gs_counts():
    return {'n_participants': ref(COHORT, 'n_participants'),
            'n_events': ref(COHORT, 'n_events_total'), 'n_endpoints': ref(COHORT, 'n_endpoints')}


def setting_fields(scenario, missing=False):
    fields = {'measurement_label': literal(SCENARIOS[scenario]),
              'data_condition': literal('Empirical complete profiles' if scenario == 'full' else
                  'Simulated missingness on empirical profiles' if missing else 'Simulated measurement noise on empirical profiles')}
    fields.update(gs_counts())
    if missing or scenario == 'full':
        masks = select(J_MASK, scenario=scenario)
        fields['mean_observed_cpgs_per_participant'] = ref(masks, 'kept_mean')
    else:
        inputs = select(M_INPUT, scenario=scenario)
        fields['mean_observed_cpgs_per_participant'] = ref(inputs, 'vocab_observed_per_participant_mean')
    return fields


def reduced_inputs():
    name = 'sf22a_reduced_inputs.csv'
    for rule in read(R_RULE):
        if rule['analysis'] != 'J':
            continue
        scenario = rule['scenario']
        rr = select(R_RULE, analysis='J', scenario=scenario)
        jr = select(J_RULE, scenario=scenario)
        comparator = rule['best_single_comparator']
        js = select(J_SUM, scenario=scenario, model=comparator)
        fields = setting_fields(scenario, missing=True)
        mark(name, jr, 'M_methylgpt', 'MethylGPT', 'Mean C-index gain above age and sex', 'C-index units',
             'M_methylgpt_ci95_low', 'M_methylgpt_ci95_high', **fields)
        mark(name, js, 'M_model', METHODS[comparator], 'Mean C-index gain above age and sex', 'C-index units',
             'M_model_ci95_low', 'M_model_ci95_high', **fields)
        fields.update(comparator_label=literal(METHODS[comparator]), passes_rule=ref(rr, 'outcome'),
                      note=literal('Best single comparator selected after the original analysis; interval does not include comparator selection.'))
        mark(name, rr, 'difference', f'MethylGPT minus {METHODS[comparator]}',
             'Difference in mean C-index gain', 'C-index units', 'ci95_low', 'ci95_high', **fields)
        # R supplies the relative gain against the revised comparator. Its own
        # selected source row supplies the stored interval for that ratio.
        mark(name, js, 'relative_gain', f'MethylGPT relative to {METHODS[comparator]}',
             'Relative gain in mean C-index gain', 'Fraction',
             'relative_gain_ci95_low', 'relative_gain_ci95_high', **fields)


def noise_cindex():
    name = 'sf22b_noise_cindex.csv'
    for scenario in NOISE_ORDER:
        fields = setting_fields(scenario)
        mr = select(M_RULE, scenario=scenario)
        mark(name, mr, 'M_methylgpt', 'MethylGPT', 'Mean C-index gain above age and sex', 'C-index units',
             'M_methylgpt_ci95_low', 'M_methylgpt_ci95_high', **fields)
        for model in ('mgpt2_combined_depthaware', 'grimage2_mean', 'dunedinpace_mean', 'raw_cpg_gpu_mean'):
            ms = select(M_SUM, scenario=scenario, model=model)
            mark(name, ms, 'M_model', METHODS[model], 'Mean C-index gain above age and sex', 'C-index units',
                 'M_model_ci95_low', 'M_model_ci95_high', **fields,
                 note=literal('Application outside the decision; Gaussian settings weight by known noise variance.' if 'depthaware' in model else ''))
        vr = select(V_C, scenario=scenario, contrast='best_single')
        comp = METHODS[vr[2]['comparator']]
        mark(name, vr, 'difference', f'MethylGPT with read-depth weighting minus {comp}',
             'Difference in mean C-index gain', 'C-index units', 'ci95_low', 'ci95_high', **fields,
             comparator_label=literal(comp), passes_rule=ref(vr, 'rule_arithmetic_for_reference'),
             note=literal('Application outside the decision. Rule undefined when comparator gain is non-positive; interval is for the difference.'))


def noise_hr():
    name = 'sf22c_noise_hazard_ratio.csv'
    for scenario in NOISE_ORDER:
        fields = setting_fields(scenario)
        base = select(O_MET, scenario=scenario, endpoint='mean_over_endpoints', model='mgpt2_combined', metric='log_hr_per_sd')
        if base[2]['suppressed'] != 'False':
            raise ValueError('Suppressed mean hazard ratio')
        mark(name, base, 'value', 'MethylGPT', 'Mean log hazard ratio per SD', 'Natural-log hazard ratio per SD',
             'ci95_low', 'ci95_high', **fields, suppressed=ref(base, 'suppressed'))
        raw = select(R_CLOCK, analysis='O', scenario=scenario, comparator_family='raw CpG')
        for model in ('mgpt2_combined_depthaware', 'grimage2_mean', 'dunedinpace_mean', raw[2]['comparator']):
            os = select(O_SUM, scenario=scenario, model=model)
            mark(name, os, 'H_model', METHODS[model], 'Mean log hazard ratio per SD', 'Natural-log hazard ratio per SD',
                 'H_model_ci95_low', 'H_model_ci95_high', **fields,
                 n_endpoints_not_estimable=ref(os, 'n_endpoints_not_estimable'),
                 note=literal('Best raw-CpG decision model selected separately in this setting.' if model == raw[2]['comparator'] else
                     'Application outside the decision; Gaussian settings weight by known noise variance.' if 'depthaware' in model else ''))
        sr = select(S_HR, analysis='O', scenario=scenario, endpoint='mean_over_endpoints', metric='log_hr_per_sd', contrast='best_single')
        comp = METHODS[sr[2]['comparator']]
        mark(name, sr, 'difference', f'MethylGPT minus {comp}', 'Difference in mean log hazard ratio per SD',
             'Natural-log hazard ratio per SD', 'ci95_low', 'ci95_high', **fields,
             comparator_label=literal(comp), suppressed=ref(sr, 'suppressed'), passes_rule=ref(sr, 'rule_arithmetic'))
        vr = select(V_H, scenario=scenario, metric='log_hr_per_sd', contrast='best_single')
        comp = METHODS[vr[2]['comparator']]
        mark(name, vr, 'difference', f'MethylGPT with read-depth weighting minus {comp}',
             'Difference in mean log hazard ratio per SD', 'Natural-log hazard ratio per SD', 'ci95_low', 'ci95_high', **fields,
             comparator_label=literal(comp), suppressed=ref(vr, 'suppressed'), passes_rule=ref(vr, 'rule_arithmetic_for_reference'),
             note=literal('Application outside the decision; rule arithmetic for reference.'))


def unmeasured_ewas():
    name = 'sf23a_ewas_unmeasured.csv'
    rows = [(select(D_RULE, setting='hm27'), 'Type 2 diabetes', 'best_comparator_value', 'best_comparator_ci95_low', 'best_comparator_ci95_high')]
    for row in read(D2_END):
        if row['setting'] == 'hm27' and row['in_primary_mean'] == 'True':
            rows.append((select(D2_END, setting='hm27', endpoint=row['endpoint']), ENDPOINTS[row['endpoint']],
                         'nearest_neighbours_value', 'nearest_neighbours_ci95_low', 'nearest_neighbours_ci95_high'))
    for row in read(D2_RULE):
        if row['setting'] == 'hm27':
            label = 'Mean over seven endpoints excluding type 2 diabetes' if row['n_endpoints'] == '7' else 'Mean over eight endpoints'
            rows.append((select(D2_RULE, setting='hm27', endpoint_set=row['endpoint_set']), label,
                         'nearest_neighbours_value', 'nearest_neighbours_ci95_low', 'nearest_neighbours_ci95_high'))
    for source, endpoint, comp, clo, chi in rows:
        record = source[2]
        fields = {'endpoint_label': literal(endpoint), 'measurement_label': literal('27K-array CpGs'),
                  'data_condition': literal('Simulated missingness on empirical profiles'),
                  'n_participants': ref(source, 'n_participants'), 'n_target_cpgs': ref(source, 'n_target_cpgs'),
                  'n_events': ref(source, 'n_events' if 'n_events' in record else 'n_events_total'),
                  'note': literal('Percentile intervals of single correlations may not contain the point estimate; use the paired-difference interval for inference.')}
        if 'n_endpoints' in record:
            fields['n_endpoints'] = ref(source, 'n_endpoints')
        mark(name, source, 'methylgpt_value', 'MethylGPT imputation', 'Spearman correlation of per-CpG Cox z with full-data z',
             'Correlation', 'methylgpt_ci95_low', 'methylgpt_ci95_high', **fields)
        mark(name, source, comp, 'Nearest-neighbour imputation', 'Spearman correlation of per-CpG Cox z with full-data z',
             'Correlation', clo, chi, **fields)
        mark(name, source, 'difference', 'MethylGPT imputation minus nearest-neighbour imputation',
             'Difference in Spearman correlation', 'Correlation units', 'difference_ci95_low', 'difference_ci95_high',
             **fields, comparator_label=literal('Nearest-neighbour imputation'), passes_rule=ref(source, 'pass'))


def noisy_ewas():
    name = 'sf23b_ewas_noise.csv'
    for rule in sorted(read(N_RULE), key=lambda r: NOISE_ORDER.index(r['scenario'])):
        scenario = rule['scenario']
        source = select(N_RULE, scenario=scenario)
        fields = {'measurement_label': literal(SCENARIOS[scenario]), 'endpoint_label': literal('Mean over eight endpoints'),
                  'data_condition': literal('Simulated measurement noise on empirical profiles'),
                  **{k: ref(source, v) for k, v in [('n_participants','n_participants'), ('n_events','n_events_total'),
                      ('n_endpoints','n_endpoints'), ('n_target_cpgs','n_target_cpgs')]},
                  'note': ref(source, 'interval_caveat')}
        for prefix, label in [('methylgpt', 'MethylGPT'), ('knn_denoised', 'Nearest-neighbour denoising'), ('raw_noisy', 'Noisy values')]:
            mark(name, source, prefix + '_value', label, 'Mean Spearman correlation of per-CpG Cox z with full-data z',
                 'Correlation', prefix + '_ci95_low', prefix + '_ci95_high', **fields)
        app = select(N_APP, scenario=scenario, application='methylgpt_depthaware', endpoint_set='eight endpoints')
        if app[2]['status'] != 'run':
            raise ValueError(f'Application not run: {scenario}')
        mark(name, app, 'value', 'MethylGPT with read-depth weighting',
             'Mean Spearman correlation of per-CpG Cox z with full-data z', 'Correlation', 'ci95_low', 'ci95_high', **fields)
        comp_labels = {'knn_denoised': 'Nearest-neighbour denoising', 'raw_noisy': 'Noisy values'}
        comp = comp_labels[rule['best_comparator']]
        mark(name, source, 'difference', f'MethylGPT minus {comp}', 'Difference in mean Spearman correlation',
             'Correlation units', 'difference_ci95_low', 'difference_ci95_high', **fields,
             comparator_label=literal(comp), passes_rule=ref(source, 'pass'))
        comp = comp_labels[app[2]['vs_best_comparator']]
        mark(name, app, 'vs_difference', f'MethylGPT with read-depth weighting minus {comp}',
             'Difference in mean Spearman correlation', 'Correlation units', 'vs_difference_ci95_low', 'vs_difference_ci95_high',
             **fields, comparator_label=literal(comp), passes_rule=ref(app, 'vs_pass'))
    OMISSIONS.append(f'Cannot proceed - required data file not found: {N_RULE} row scenario=depth_1 (also absent from N_applications.csv); omitted four method estimates and two best-comparator differences.')


def individual_diseases():
    name = 'sf24_individual_diseases.csv'
    labels = {'COPD': 'Chronic obstructive pulmonary disease', 'Colon_Rectum_cancer': 'Colorectal cancer'}
    for clock, contrast in [('DunedinPACE', 'U_D: MethylGPT added to DunedinPACE'), ('GrimAge2', 'U_G: MethylGPT added to GrimAge2')]:
        for row in read(U_DISEASE):
            if row['setting'] != 'full' or row['contrast'] != contrast:
                continue
            source = select(U_DISEASE, setting='full', contrast=contrast, disease=row['disease'])
            meta = select(K_DISEASE, constituent=row['disease'])
            disease = labels.get(row['disease'], row['disease'].replace('_', ' ').capitalize())
            mark(name, source, 'difference', f'MethylGPT + {clock} minus {clock} alone', 'Difference in C-index',
                 'C-index units', 'ci95_low', 'ci95_high', disease_label=literal(disease),
                 measurement_label=literal('Complete profiles'), data_condition=literal('Empirical complete profiles'),
                 n_participants=ref(meta, 'n_at_risk'), n_events=ref(source, 'events'),
                 q_value=ref(source, 'q_bh_across_diseases'), p_value=ref(source, 'p_boot'),
                 comparator_label=literal(f'{clock} alone'),
                 note=literal('Benjamini-Hochberg q across 13 diseases within this contrast.'))
        source = select(U_MEAN, setting='full', contrast=contrast)
        k = select(K_RULE, setting='full', statistic='mean')
        mark(name, source, 'difference', f'MethylGPT + {clock} minus {clock} alone', 'Mean difference in C-index over diseases',
             'C-index units', 'ci95_low', 'ci95_high', disease_label=literal('Mean over 13 diseases'),
             measurement_label=literal('Complete profiles'), data_condition=literal('Empirical complete profiles'),
             n_participants=ref(COHORT, 'n_participants'), n_diseases=ref(k, 'n_diseases'),
             p_value=ref(source, 'p_boot'), passes_rule=ref(source, 'outcome'),
             comparator_label=literal(f'{clock} alone'), note=literal('Unweighted disease mean; no mean-row q value stored.'))


def array_agreement():
    name = 'sf25a_array_agreement.csv'
    quantities = {'GrimAge2': 'GrimAge2', 'DunedinPACE': 'DunedinPACE', 'PhenoAge': 'PhenoAge',
                  'age': 'Age', 'Horvathv1': 'Horvath age',
                  'primary (GrimAge2, DunedinPACE)': 'Primary mean of GrimAge2 and DunedinPACE'}
    for row in read(C_PER):
        if row['method'] not in ('MethylGPT', 'R_biolearn'):
            continue
        source = select(C_PER, comparison_id=row['comparison_id'], quantity=row['quantity'], method=row['method'])
        mark(name, source, 'D', 'MethylGPT' if row['method'] == 'MethylGPT' else 'Raw clocks (reference filling)',
             'Mean absolute between-array difference on own SD', 'Own-method array-A SD units',
             quantity_label=literal(quantities[row['quantity']]),
             comparison_label=literal(row['gse'] + ' ' + row['comparison'].replace('-', ' versus ')),
             aggregation_label=literal('Individual comparison'), n_pairs=ref(source, 'n_pairs'),
             data_condition=literal('Paired empirical array profiles'),
             note=literal('No per-comparison interval stored. Age and Horvath age are separate source quantities.'))
    for row in read(C_SUM):
        if row['metric'] != 'D':
            continue
        source = select(C_SUM, subset=row['subset'], quantities=row['quantities'], metric='D')
        fields = {'quantity_label': literal(quantities[row['quantities']]),
                  'comparison_label': literal('All six comparisons' if row['subset'] == 'all comparisons' else 'Three comparisons without profiles in the corpus training split'),
                  'aggregation_label': literal('Unweighted mean over comparisons'),
                  'n_comparisons': ref(source, 'comparisons'), 'data_condition': literal('Paired empirical array profiles')}
        for col, label in [('value_MethylGPT', 'MethylGPT'), ('value_R_biolearn', 'Raw clocks (reference filling)')]:
            mark(name, source, col, label, 'Mean absolute between-array difference on own SD',
                 'Own-method array-A SD units', **fields,
                 note=literal('Mean over comparisons, not a participant-pooled estimate; method-specific absolute intervals not stored.'))
        comp = 'Raw clocks (reference filling)' if row['best_comparator'] == 'R_biolearn' else 'Raw clocks (mean filling)'
        mark(name, source, 'difference_methylgpt_minus_best', f'MethylGPT minus {comp}',
             'Difference in mean absolute between-array difference', 'Own-method array-A SD units',
             'difference_ci95_low', 'difference_ci95_high', **fields, comparator_label=literal(comp))
        mark(name, source, 'relative_gain', f'MethylGPT relative to {comp}', 'Relative reduction in between-array difference',
             'Fraction', 'relative_gain_ci95_low', 'relative_gain_ci95_high', **fields, comparator_label=literal(comp))
    OMISSIONS.append(f'Cannot proceed - required data file not found: {C_PER} quantity=primary (GrimAge2, DunedinPACE); omitted 12 per-comparison primary-mean marks (six comparisons, two methods). Only stored across-comparison primary means are included.')


def sparse_labels():
    name = 'sf25b_few_labels_sparse.csv'
    methods = {'imputed': 'MethylGPT-imputed CpGs', 'raw_knn': 'Nearest-neighbour imputation',
               'raw_mean': 'Raw CpGs (mean imputation)', 'clocks': 'Clocks'}
    results = read(G_JSON)
    for row in read(G_SUM):
        if row['set'] != 'primary' or row['methylgpt_representation'] != 'imputed':
            continue
        summary = select(G_SUM, set='primary', f=row['f'], n_train=row['n_train'], methylgpt_representation='imputed')
        key = f"primary|f={row['f']}|n={row['n_train']}|imputed"
        if key == results['primary']['set']:
            record, pointer = results['primary'], ['primary']
        else:
            record, pointer = results['secondary'][key], ['secondary', key]
        source = (G_JSON, {'json_path': pointer}, record)
        fields = {'aggregation_label': literal('Mean over held-out datasets'), 'keep_fraction': ref(summary, 'f'),
                  'n_train': ref(summary, 'n_train'), 'n_datasets': ref(source, 'datasets'),
                  'data_condition': literal('Simulated missingness on empirical profiles')}
        for model, label in methods.items():
            mark(name, source, ('M', model), label, 'Mean AUC minus 0.5', 'AUC units',
                 ('M_ci95', model, 0), ('M_ci95', model, 1), **fields)
        for model in ('raw_knn', 'raw_mean', 'clocks'):
            mark(name, source, ('vs_each', model, 'difference'), f'MethylGPT-imputed CpGs minus {methods[model]}',
                 'Difference in mean AUC', 'AUC units', ('vs_each', model, 'ci95', 0), ('vs_each', model, 'ci95', 1),
                 **fields, comparator_label=literal(methods[model]),
                 note=literal('Nearest-neighbour imputation is the best comparator.'),
                 **({'passes_rule': ref(source, 'pass')} if model == 'raw_knn' else {}))
    for row in read(G_PER):
        if row['f'] != '0.05' or row['n_train'] != '50' or row['representation'] not in ('imputed', 'raw_knn'):
            continue
        source = select(G_PER, f='0.05', n_train='50', accession=row['accession'], representation=row['representation'])
        data = select(G_DATA, set='heldout', accession=row['accession'])
        mask = select(G_MASK, set='heldout', accession=row['accession'], f='0.05')
        mark(name, source, 'auc_mean', methods[row['representation']], 'Per-dataset mean AUC', 'AUC',
             aggregation_label=literal('Individual held-out dataset'), dataset_label=ref(source, 'accession'),
             disease_label=ref(source, 'disease'), array_label=ref(source, 'array'),
             keep_fraction=ref(source, 'f'), n_train=ref(source, 'n_train'),
             n_cases=ref(data, 'n_case'), n_controls=ref(data, 'n_ctrl'), n_profiles=ref(mask, 'profiles'),
             mean_observed_cpgs_per_participant=ref(mask, 'mean_observed'),
             mean_kept_cpgs_per_participant=ref(mask, 'mean_kept'), auc_sd=ref(source, 'auc_sd'),
             data_condition=ref(source, 'masking'),
             note=literal('Per-dataset values are AUC; summary values are AUC minus 0.5. auc_sd is the stored draw SD, not a 95% interval.'))


def verify_and_write():
    expected_counts = {
        'sf22a_reduced_inputs.csv': 28, 'sf22b_noise_cindex.csv': 54,
        'sf22c_noise_hazard_ratio.csv': 63, 'sf23a_ewas_unmeasured.csv': 30,
        'sf23b_ewas_noise.csv': 30, 'sf24_individual_diseases.csv': 28,
        'sf25a_array_agreement.csv': 108, 'sf25b_few_labels_sparse.csv': 111,
    }
    for name, rows in TABLES.items():
        assert len(rows) == expected_counts[name], (name, len(rows))
        for row, origins in zip(rows, TRACE[name]):
            for field, origin in origins.items():
                if 'json_path' in origin['filter']:
                    record = nested(read(origin['file']), origin['filter']['json_path'])
                else:
                    record = select(origin['file'], **origin['filter'])[2]
                assert row[field] == nested(record, origin['column']), (name, field, origin)
            if row['ci95_low'] != '' and row['ci95_high'] != '':
                assert float(row['ci95_low']) <= float(row['ci95_high'])
        columns = list(dict.fromkeys(key for row in rows for key in row))
        # Leave provenance at the end even when later rows introduce fields.
        columns = [k for k in columns if not k.startswith('source_')] + [k for k in columns if k.startswith('source_')]
        with (OUT / name).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        saved = list(csv.DictReader((OUT / name).open()))
        assert len(saved) == len(rows)
        for original, restored in zip(rows, saved):
            assert all(restored.get(k, '') == ('' if v is None else str(v)) for k, v in original.items())
        print(f'{name}: shape=({len(rows)}, {len(columns)})')
    sources = {}
    for path in sorted(CACHE):
        file = ROOT / path
        sources[path] = {'sha256': hashlib.sha256(file.read_bytes()).hexdigest(),
                         'mtime_utc': datetime.fromtimestamp(file.stat().st_mtime, timezone.utc).isoformat(),
                         'bytes': file.stat().st_size}
    (OUT / 'source_provenance_2026-11-10.json').write_text(json.dumps({'sources': sources, 'rows': TRACE}, indent=2, ensure_ascii=False) + '\n')
    report = ['# Supplementary figure table report', '',
              'All plotted values and stored interval limits match their aggregate source cells exactly. '
              'The provenance file records each copied field and source-file timestamps and checksums. '
              'Only selection, readable labels, metadata joins and reshaping were performed.', '',
              f'The eight CSVs contain {sum(len(rows) for rows in TABLES.values())} marks from {len(sources)} aggregate source files. '
              'Validation checked each copied field against its source, CSV round trips, finite plotted values, '
              'ordered interval limits and expected table sizes. The README gives units, sample counts and legend requirements.', '',
              '## Unavailable rows', '', *OMISSIONS, '', '## CSV previews', '']
    for name, rows in TABLES.items():
        with (OUT / name).open() as handle:
            reader = csv.reader(handle)
            preview = [next(reader) for _ in range(min(6, len(rows) + 1))]
        buf = io.StringIO()
        csv.writer(buf, lineterminator='\n').writerows(preview)
        report.extend([f'### {name}', '', f'Shape: ({len(rows)}, {len(preview[0])}). Head: first five rows.', '', '```csv', buf.getvalue().rstrip(), '```', ''])
    (OUT / 'REPORT.md').write_text('\n'.join(report))


def main():
    try:
        reduced_inputs()
        noise_cindex()
        noise_hr()
        unmeasured_ewas()
        noisy_ewas()
        individual_diseases()
        array_agreement()
        sparse_labels()
        verify_and_write()
    except (OSError, ValueError, KeyError, AssertionError) as exc:
        raise SystemExit(f'Table assembly failed: {exc}') from exc


if __name__ == '__main__':
    main()
