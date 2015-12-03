package main;

import java.util.ArrayList;
import java.util.Arrays;
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
		Instances ins = Lector.getLector().leerInstancias("colon.arff");
		
		// Copiamos el mismo archivo pero sin la clase
		Remove rm = new Remove();
		rm.setAttributeIndicesArray(new int[]{ins.classIndex()});
		rm.setInputFormat(ins);
		Instances ins2 = Filter.useFilter(ins, rm);
		
		//parametros del cluster
		int k = 2;
		//long tiempo= 1000 * 2;
		
		// Generamos el Clusterizador
		Kmeans kmeans = new Kmeans(k,2);
		//kmeans.setTimeOut(tiempo);
		double mediaSSE;
		double mediaShilouette;
		double repeticiones = 10;
		long tI, tF, mediaT;
		int mediaIt;
		System.out.println("K\tSSE\tSilhouette\tTiempo\tIteraciones");
		for (int K=2; K<=50; K++){
			kmeans.setClusters(K);
			mediaSSE=0;
			mediaShilouette=0;
			mediaT = 0;
			mediaIt = 0;
			for (int repeticion=0; repeticion<repeticiones; repeticion++){
				tI = System.currentTimeMillis();
				kmeans.buildClusterer(ins2);
				tF = System.currentTimeMillis();
				
				mediaIt += kmeans.getIterationsDone();
				mediaT += tF-tI;
				mediaSSE += kmeans.SSE();
				mediaShilouette += kmeans.silhouette();
			}
			System.out.println(K+"\t"+(""+(mediaSSE/repeticiones)).replace(".", ",")+
					"\t"+(""+(mediaShilouette/repeticiones)).replace(".", ",")+
					"\t"+((mediaT/repeticiones)+"").replace(".", ",")+
					"\t"+((mediaIt/repeticiones)+"").replace(".", ","));
		}
		
		
		
		//Mi metodo
		/*
		boolean encontrado;
		ArrayList<int[][]> todos = new ArrayList<int[][]>();
		class Numero {
			private double num;
			Numero(double n){
				this.num = n;
			}
			public double getNum() {
				return num;
			}
			public void setNum(double num) {
				this.num = num;
			}
		}
		ArrayList<Numero[]> cantidad = new ArrayList<Numero[]>();
		int i2 =0;
		int max = 50;
		for (int i = 0; i < max; i++) {
			if (i2<(int)((i*10)/max)){
				i2 =(int)((i*10)/max);
				System.out.print("#");
			}
			kmeans.buildClusterer(ins2);
			int[][] ma = Ejecutador.getMatrix(ins, kmeans);
			Ejecutador.ordenar(ma);
			encontrado = false;
			Iterator<Numero[]> it = cantidad.iterator();
			for (int[][] bi : todos) {
				Numero s = it.next()[0];
				if (Ejecutador.comprobarIguales(ma, bi)){
					s.setNum(s.getNum()+1);
					encontrado = true;
					break;
				}
			}
			if (!encontrado){
				todos.add(ma);
				cantidad.add(new Numero[]{new Numero(1), new Numero(kmeans.silhouette()), new Numero(kmeans.SSE())});
			}
			//System.out.println(Ejecutador.impMatrix(ma, ins));
		}
		Iterator<Numero[]> it = cantidad.iterator();
		System.out.println();
		for (int[][] is : todos) {
			Numero[] s = it.next();
			System.out.println("################################################################");
			System.out.println(s[0].getNum());
			System.out.println("Silhouette: "+s[1].getNum());
			System.out.println("SSE: "+s[2].getNum());
			System.out.println(Ejecutador.impMatrix(is, ins));
		}*/
	}
	private static boolean comprobarIguales(int[][] uno, int[][] dos){
		if (uno.length != dos.length) return false;
		for (int i = 0; i < uno.length; i++) {
			int[] bat = uno[i];
			int[] bi = dos[i];
			if (bat.length!=bi.length) return false;
			for (int j = 0; j < bat.length; j++) {
				if(bat[j]!=bi[j]) return false;
			}
		}
		return true;
	}
	
	private static void ordenar(int[][] ma) {
		int mayor=-1;
		int num = 0;
		for (int i = 0; i < ma.length; i++) {
			int[] fila = ma[i];
			if (i>=fila.length) break;
			for (int j = i; j < fila.length; j++) {
				int r = fila[j];
				if (r>mayor){
					num = j;
					mayor = r;
				}
			}
			Ejecutador.movercl(ma, i, num);
			num = 0;
			mayor = -1;
		}
	}
	private static void movercl(int[][] ma, int a, int b){
		for (int[] is : ma) {
			int tmp = is[a];
			is[a] = is[b];
			is[b] = tmp;
		}
	}
	private static void moverln(int[][] is, int a, int b){
		int[] tmp = is[a];
		is[a] = is[b];
		is[b] = tmp;
	}

	private static int[][] getMatrix(Instances ins, Kmeans kmeans) throws Exception{
		int k = kmeans.numberOfClusters();
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
		return grupos;
	}
	private static String impMatrix(int[][] grupos, Instances ins){
		String result="";
		result += ("------------Matriz De Confusión-------\n");
		result += ("\n");
		result += "\t";
		for (int i = 0; i < grupos[0].length; i++) {
			result +=("  "+(char)(97+i) + "\t");
		}
		result+="   <-- Clusterizado como\n";
		result += "\t";
		for (int i = 0; i < grupos[0].length; i++) {
			result +=("+-------");
		}
		result+="+\n";
		Enumeration<Object> clases = ins.classAttribute().enumerateValues();
		for (int f = 0; f < grupos.length; f++) {
			result += "\t";
			for (int i = 0; i < grupos[0].length; i++) {
				result +=("| "+grupos[f][i] + "\t");
			}
			result+="| "+clases.nextElement()+"\n";
			result += "\t";
			for (int i = 0; i < grupos[0].length; i++) {
				result +=("+-------");
			}
			result+="+\n";
		}
		return result;
	}
}
