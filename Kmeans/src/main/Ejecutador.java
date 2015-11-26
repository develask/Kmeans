package main;

import java.util.ArrayList;
import java.util.Enumeration;
import java.util.Iterator;

import src.Kmeans;
import src.Lector;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class Ejecutador {
	public static void main(String[] args) throws Exception {
		//Cargamos el iris.arff
		Instances ins = Lector.getLector().leerInstancias("iris.arff");
		
		// Copiamos el mismo archivo pero sin la clase
		Remove rm = new Remove();
		rm.setAttributeIndicesArray(new int[]{ins.classIndex()});
		rm.setInputFormat(ins);
		Instances ins2 = Filter.useFilter(ins, rm);
		
		//parametros del cluster
		int k = 3;
		long tiempo= 1000 * 20;
		
		// Generamos el Clusterizador
		Kmeans kmeans = new Kmeans(k,2);
		kmeans.setTimeOut(tiempo);
		
		ArrayList<Object[]> results = new ArrayList<Object[]>();
		double num;
		boolean enc;
		for (int i = 0; i < 50; i++) {
			System.out.print(i + " ");
			kmeans.buildClusterer(ins2);
			num = kmeans.SSE();
			enc = false;
			for (Object[] objects : results) {
				if (objects[0].equals(num)){
					enc = true;
					break;
				}
			}
			if (!enc){
				results.add(new Object[]{num,kmeans.distributionForCluster()});
			}
		}
		System.out.println();
		for (Object[] objects : results) {
			System.out.println("----------------");
			System.out.println("Silhouette: " + objects[0]);
			int i = 0;
			for (ArrayList<Instance> instances : (ArrayList<Instance>[])objects[1]) {
				System.out.println("\t- Grupo "+ ++i + ":\t" + instances.size() + " instancias.");
			}
		}
		
		
		/*
		//Clasificar
		int[][] grupos = new int[ins.classAttribute().numValues()][k];
		for (int i = 0; i < grupos.length; i++) {
			int[] js = grupos[i];
			for (int j = 0; j < js.length; j++) {
				js[j]= 0;
				
			}
		}
		for (Instance instance : ins) {
			grupos[(int)instance.classValue()][kmeans.clusterInstance(instance)]++;
		}
		String result = "";
		result += ("------------Matriz De Confusión-------\n");
		result += ("\n");
//		result += "\t";
//		for (int i = 0; i < grupos.length; i++) {
//			result +=("+-------");
//		}
//		result+="\n";
		result += "\t";
		for (int i = 0; i < grupos.length; i++) {
			result +=("  "+(char)(97+i) + "\t");
		}
		result+="   <-- Clusterizado como\n";
		result += "\t";
		for (int i = 0; i < grupos.length; i++) {
			result +=("+-------");
		}
		result+="+\n";
		Enumeration<Object> clases = ins.classAttribute().enumerateValues();
		for (int f = 0; f < grupos.length; f++) {
			result += "\t";
			for (int i = 0; i < grupos.length; i++) {
				result +=("| "+grupos[f][i] + "\t");
			}
			result+="| "+clases.nextElement()+"\n";
			result += "\t";
			for (int i = 0; i < grupos.length; i++) {
				result +=("+-------");
			}
			result+="+\n";
		}
		System.out.println(result);*/
	}
}
