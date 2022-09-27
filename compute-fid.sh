TEMPPATH=./.temp  # temp folder to store the data
OUTPATH=/home/jeanner211/RESULTS/ACCV-Rebuttal/l1-perc0/q-31_t--1
EXPPATH=l1-0.05

mkdir -p ${TEMPPATH}/real
mkdir -p ${TEMPPATH}/cf
mkdir -p ${TEMPPATH}/cfmin

echo 'Copying CF images '

cp -r ${OUTPATH}/Results/${EXPPATH}/CC/CCF/CF/* ${TEMPPATH}/cf
cp -r ${OUTPATH}/Results/${EXPPATH}/IC/CCF/CF/* ${TEMPPATH}/cf

echo 'Copying real images'

cp -r ${OUTPATH}/Original/Correct/* ${TEMPPATH}/real
cp -r ${OUTPATH}/Original/Incorrect/* ${TEMPPATH}/real

echo 'Computing FID'

python -m pytorch_fid ${TEMPPATH}/real ${TEMPPATH}/cf --device cuda:0

rm -rf ${TEMPPATH}
