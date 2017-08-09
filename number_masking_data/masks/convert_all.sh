for file in *.png
do
  echo "processing $file"
  convert $file -brightness-contrast 0x100 -negate $file
done
