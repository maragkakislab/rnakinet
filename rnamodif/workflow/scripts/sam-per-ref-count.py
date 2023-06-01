#!/usr/bin/env python
import pysam
import argparse

parser = argparse.ArgumentParser(description='Count reads aligned on reference sequence in a SAM/BAM file')
parser.add_argument("-i", "--ifile", help="Input SAM/BAM file")
parser.add_argument("-s", "--sam", action='store_true', help="Input file format only if SAM file; default BAM format")
parser.add_argument("-r", "--ref-col-name", default="reference", help="Name of output column with reference ids, default: reference")
parser.add_argument("-c", "--cnt-col-name", default="count", help="Name of output column with read count, default: count")
parser.add_argument("-n", "--opt-col-name", help="Name of an optional column e.g. sample_name")
parser.add_argument("-v", "--opt-col-val", help="Value for the optional column; same for all rows")
parser.add_argument("-d", "--delim", default="\t", help="Delimiter to separate the columns of the output file, default : TAB")
parser.add_argument("-x", "--no-zeros", action='store_true', help="Skip references with 0 reads")
args = parser.parse_args()

ifiletype = "rb"
if args.sam:
    ifiletype = "r"

bamfile = pysam.AlignmentFile(args.ifile, ifiletype)

reference_counts = {}
for seq in bamfile:
    if seq.is_unmapped:
        continue

    reference = seq.reference_name
    query_length = seq.query_length
    if reference not in reference_counts:
        reference_counts[reference] = 0
    reference_counts[reference] += 1

# If transcripts with 0 counts are not explicitly excluded, then add them to
# the dictionary. Assumes that the SAM header exists.
if not args.no_zeros:
    header = bamfile.header
    if 'SQ' in header:
        for elem in header['SQ']:
            ref = elem['SN']
            if ref not in reference_counts:
                reference_counts[ref] = 0

delim = args.delim

#print the header of file
header = [args.ref_col_name, args.cnt_col_name]
if args.opt_col_name and args.opt_col_val:
    header += [args.opt_col_name]
print(delim.join(header))

#print the contents of file
for reference, count in reference_counts.items():
    row = [reference, str(count)]
    if args.opt_col_name and args.opt_col_val:
        row += [args.opt_col_val]
    print(delim.join(row))
