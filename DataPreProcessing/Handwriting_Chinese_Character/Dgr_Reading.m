function Dgr_Reading(src_path,save_path_prefix,num_writer_each_group)


file_list = dir([src_path,'*.dgr']);
last_writer_id=file_list(length(file_list)).name;
last_writer_id=strfind(last_writer_id,'-');
last_writer_id=str2num(file_list(length(file_list)).name(1:last_writer_id-1));

full_counter=0;
full_valid_counter=0;
previous_writer=-1;
for ii = 1:length(file_list)
    writer_id=strfind(file_list(ii).name,'-');
    writer_id=str2num(file_list(ii).name(1:writer_id-1));
    
    if writer_id~=previous_writer
        previous_writer=writer_id;
        character_in_one_writer_counter=0;
    end
    
    
    if ii==1 || writer_id>next_writer_id
        prev_writer_id=writer_id;
        next_writer_id=writer_id+num_writer_each_group-1;
        if next_writer_id>last_writer_id
            next_writer_id=last_writer_id;
        end
        
        save_path_prefix_current=sprintf([save_path_prefix,'writer%05d_To_%05d/'],prev_writer_id,next_writer_id);
        mkdir(save_path_prefix_current);
        
        
        disp(save_path_prefix)
    end
    
    [thiswriter_valid, thiswriter_full,character_in_one_writer_counter]=Dgr_Reading_Implementation([src_path,file_list(ii).name],save_path_prefix_current,character_in_one_writer_counter);
    full_counter = full_counter + thiswriter_full;
    full_valid_counter = full_valid_counter +thiswriter_valid;
    disp(sprintf([src_path,file_list(ii).name,': %d/%d, FullValid/ThisWriterValid:%d/%d, Full/ThisWriterFull:%d/%d'],ii,length(file_list),full_valid_counter,thiswriter_valid,full_counter,thiswriter_full))

end



end







function [valid_counter,full_counter,character_in_one_writer_counter]=Dgr_Reading_Implementation(input_path,save_prefix_64,character_in_one_writer_counter)

fid=fopen(input_path);
file_name_start_index=strfind(input_path,'/');
file_name_start_index=file_name_start_index(length(file_name_start_index))+1;
file_name_end_index=strfind(input_path,'.dgr');
file_name_end_index=file_name_end_index(length(file_name_end_index))-1;
file_name=input_path(file_name_start_index:file_name_end_index);
split_index=strfind(file_name,'-');
writerId=file_name(1:split_index-1);
writerId=str2num(writerId);

headerSize=fread(fid,1,'uint32');
illustrationLength=headerSize-36;

formatCodeAscii=fread(fid,8,'uchar');
formatCode=char(formatCodeAscii)';

illustrationAscii=fread(fid,illustrationLength,'uchar');
illustration=char(illustrationAscii)';

codeTypeAscii=fread(fid,20,'uchar');
codeType=(char(codeTypeAscii))';

codeLength=fread(fid,1,'uint16');

bitsperpix=fread(fid,1,'uint16');


im_h=fread(fid,1,'uint32');
im_w=fread(fid,1,'uint32');
line_n=fread(fid,1,'uint32');


valid_counter=0;
full_counter=0;
for i=1:line_n
    char_n{i}=fread(fid,1,'uint32');


    iii=1;
    for ii=1:char_n{i}

        classLabel_curt=fread(fid,2,'uchar');
        topleft_c=fread(fid,2,'uint16');
        h=fread(fid,1,'uint16');
        w=fread(fid,1,'uint16');
        size=h*w;
        bitmap_curt=fread(fid,size,'uchar');
        bitmap_curt=(reshape(bitmap_curt,[w h]))';
        bitmap_curt=imresize(bitmap_curt,[150,150]);
%         check1=find(bitmap_curt>240);
%         check2=find(bitmap_curt<=240);
%         bitmap_curt(check1)=255;
%         bitmap_curt(check2)=0;
        
        bitmap_curt=bitmap_curt/255;
        
        output_pict_256 = ones([256,256]);
        output_pict_256(52:52+149,52:52+149)=bitmap_curt;
        
        
        

        if classLabel_curt(1)>=176 && classLabel_curt(1)~=255

            class_label_formatted = sprintf('%03d%03d',classLabel_curt(1),classLabel_curt(2));
            writer_id_formatted =sprintf('%05d',writerId);
            counter_formatted=sprintf('%09d',character_in_one_writer_counter);
            

            
            output_pict_64 = imresize(output_pict_256,[64,64]);
            save_file_name_64=[save_prefix_64,counter_formatted,'_',class_label_formatted,'_',writer_id_formatted,'.png'];
            imwrite(output_pict_64,save_file_name_64)
 
            
            
            valid_counter = valid_counter +1;
            character_in_one_writer_counter = character_in_one_writer_counter +1;
        end
        full_counter = full_counter + 1;


    end
end
sta=fclose(fid);
 
end

