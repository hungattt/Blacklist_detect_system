import customAxios from "../utils/customAxios";

type RequestOptions = {
  headers: {
    "Content-Type": string;
  };
};

type postDataSupervisionStreamingService = {
  camera: string;
  video?: File | string;
};

export const postData = async ({
  data,
  options,
}: {
  data: postDataSupervisionStreamingService;
  options: RequestOptions;
}) => {
  try {
    const response = await customAxios.post(
      "facialrecognition/",
      data,
      options
    );
    return response.data;
  } catch (error) {
    console.error("FacialRecognitionServiceError: ", error);
  }
};
