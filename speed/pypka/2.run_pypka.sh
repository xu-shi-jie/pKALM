cd speed/samples
# iterate over all PDB files in the directory
for file in *.pdb; 
do
    stem=$(basename $file .pdb)
    if [ -f ${stem}.pka ]; then
        echo "File ${stem}.pka already exists. Skipping..."
        continue
    fi
    
    # check if stem[:4] is in the list of PDBs that failed to run
    if grep -q ${stem:0:4} ../failed_pypka.txt; then
        echo "Skipping $stem"
        continue
    fi

    echo "Running pypka on $file"
    echo "structure       = $stem.pdb      # Input structure file name
ncpus           = -1            # Number of processes (-1 for maximum allowed)
epsin           = 15            # Dielectric constant of the protein
ionicstr        = 0.1           # Ionic strength of the medium (M)
pbc_dimensions  = 0             # PB periodic boundary conditions (0 for solvated proteins and 2 for lipidic systems)
sites           = all           # Titrate all available sites
output          = $stem.pka      # pKa values output file" > parameters.txt
    docker run -v ${PWD}:/home/ -w /home -t pedrishi/pypka:latest timeout 400s python3.9 -m pypka parameters.txt
done