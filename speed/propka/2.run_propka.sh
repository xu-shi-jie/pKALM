cd speed/samples
# iterate over all PDB files in the directory
for file in *.pdb; 
do
    stem=$(basename $file .pdb)
    if [ -f ${stem}.pka ]; then
        echo "File ${stem}.pka already exists. Skipping..."
        continue
    fi
    echo "Running propka on $file"
    python3 -m propka $file
done