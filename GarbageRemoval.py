#Sadab Hafiz and Zachary Motassim
#This file contains functions to delete unnecessary text files, lines from text files, and sub folders from the corpora directory

#Import the necessary modules and implementation files
import os
import string
from langdetect import detect    


#Removes empty sub folders in curr_working_directory\folder_name
#@folder_name is a string
def RemoveEmptySubFolders(folder_name):
    file_path=os.getcwd()+"/"+folder_name
    removed_count=0
    subdirs=os.listdir(file_path)
    for subdir in subdirs:
        subdir_path=file_path+"/"+str(subdir)
        try:
            if(len(os.listdir(subdir_path))==0):
                os.rmdir(subdir_path)
                removed_count+=1
        except:
            print("No Subdirectories found")
            return
    print("Removed",removed_count,"folders.")

#Removes empty text files in current_working_directory\folder_name
#@folder_name is a string
def RemoveEmptyFiles(folder_name):
    file_path=os.getcwd()+"/"+folder_name
    removed_count=0
    for root, subdir, file_names in os.walk(file_path):
        for file in file_names:
            with open(os.path.join(root, file),'r+',encoding='utf8') as file_df:
                file_content=file_df.readlines()
                file_df.close()
                if(len(file_content)<=1):
                    os.remove(os.path.join(root, file))
                    removed_count+=1
            
    print("Removed",removed_count,"text files.")

#Removes text files with language except specified language from current_working_directory/folder_name
#@folder_name is a string
#@language is a two word abbreviation for the language that the user wants to keep. The one we used is "en" which stands for English
def RemoveLanguageExcept(folder_name,language):
    if(len(language)==0):
        print("No language specified")
        return
    file_path=os.getcwd()+"/"+folder_name
    removed_count=0
    for root, subdir, file_names in os.walk(file_path):
        for file in file_names:
            with open(os.path.join(root, file),'r',encoding='utf8') as file_df:
                file_content=file_df.readlines()
                file_df.close()
                file_string=""
                for line in file_content:
                    file_string+=line
                lang=detect(file_string)
                if lang!=language:
                    os.remove(os.path.join(root, file))
                    removed_count+=1
    print("Removed",removed_count,"files that are in other languages.")

#Calls the functions implemented in this file on current_working_directory/folder_name
#@folder_name is a string
#@language is the language the user wants to keep in the corpora
def FixFolder(folder_name,language):
    RemoveEmptyFiles(folder_name)
    RemoveLanguageExcept(folder_name,language)
    RemoveEmptySubFolders(folder_name)

