package src;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.ObjectInputStream;

import weka.classifiers.Classifier;
import weka.core.Instances;




public class Lector {
	
	private static Lector miLector=null;
	
	private Lector(){
		
	}
	
	public static Lector getLector(){
		if(Lector.miLector==null){
			Lector.miLector = new Lector();
		}
		return Lector.miLector;
	}
	
	public Instances leerInstancias(String path){
		// 1.2. Open the file
	    FileReader fi = null;
	    try {
			fi=new FileReader(path);
	    }catch (FileNotFoundException e) {
			System.out.println("ERROR: Revisa el path del fichero:"+path);
			return null;
		}         
	    // 1.3. Load the instances
		Instances data=null;
		try {
			data = new Instances(fi);
		} catch (IOException e) {
			System.out.println("ERROR: Revisa el contenido del fichero: "+path);
			return null;
		}
	  	// 1.4. Close the file
		try {
			fi.close();
		} catch (IOException e) {
			return null;
		}
		// 1.6. Specify which attribute will be used as the class: the last one, in this case 
		data.setClassIndex(data.numAttributes()-1);      
	    return data;
	}
	public Classifier cargarModelo(String modeloPath){
		ObjectInputStream ois;
		Classifier cls = null;
		try {
			ois = new ObjectInputStream(new FileInputStream(modeloPath));
			cls = (Classifier) ois.readObject();
			ois.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} catch (ClassNotFoundException e) {
		e.printStackTrace();
		}
		return cls;
	}
	

}
