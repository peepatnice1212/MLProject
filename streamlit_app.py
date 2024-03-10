import streamlit as st
import keras_global_wheat_detection_with_mask_rcnn

def main():
    st.title('My Streamlit Web App')
    keras_global_wheat_detection_with_mask_rcnn.my_function()

if __name__ == '__main__':
    main()
