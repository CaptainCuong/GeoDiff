import os

file_dir = ['./models', './utils', './configs']
keyword = '.*Chem*'

for dir_ in file_dir:
	print('\n','-'*25,dir_,'-'*25,'\n')
	os.system('wsl -e sh -c "grep -worne \'' + keyword + '\' ' + dir_ + '"')
	print('\n','*-'*25,'\n')
os.system('wsl -e sh -c "grep -wone \'' + keyword + '\' ' + '* .*"')