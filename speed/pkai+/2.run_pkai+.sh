cd speed/samples
# iterate over all PDB files in the directory
for file in *.pdb; 
do
    stem=$(basename $file .pdb)
    if [ -f ${stem}.pka ]; then
        echo "File ${stem}.pka already exists. Skipping..."
        continue
    fi
    echo "Running pkai on $file"
    pKAI $file --model pKAI+ > ${stem}.pka
done