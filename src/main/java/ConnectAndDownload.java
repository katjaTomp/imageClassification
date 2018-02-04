import com.flickr4java.flickr.*;
import com.flickr4java.flickr.photos.*;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class ConnectAndDownload {

    public ConnectAndDownload() {
        try {


            String apiKey = "01afb14873dabdec2415b14c14d4ecee";
            String secret = "1bad015b1cf00591";

            Flickr flickr = new Flickr(apiKey, secret, new REST());

            SearchParameters searchParameters = new SearchParameters();

            //searchParameters.setBBox("-180","-90","180","90");

            searchParameters.setMedia("photos");
            String[] tags = {"cat"};
            searchParameters.setTags(tags);

            PhotosInterface pi = new PhotosInterface(apiKey,secret,new REST());
            PhotoList<Photo> list = pi.search(searchParameters,100,1);

            System.out.println("Image List");
            for(int i=0; i< list.size(); i++){
                Photo photo = list.get(i);

                System.out.println("Image:"+ i + "\nTitle:"+ photo.getTitle()+ "\nTags:"+photo.getTags()+"\nUrl:"+
                photo.getUrl());

                // Save the image
                BufferedImage  bufferedImage = pi.getImage(photo, Size.MEDIUM);
                File outputFile = new  File("cat-"+i);
                ImageIO.write(bufferedImage,"jpg",outputFile);


            }
        }


        catch (FlickrException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }


    }


    public static void main(String[] args){

        new ConnectAndDownload();
    }


}
